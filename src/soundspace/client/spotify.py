from __future__ import annotations

import asyncio
import base64
from time import monotonic
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API_BASE = "https://api.spotify.com/v1"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_MAX_CONCURRENCY = 8
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF = 1.0
_TOKEN_EXPIRY_BUFFER = 60.0
_MAX_LIMIT = 50
_TIME_RANGES = frozenset({"short_term", "medium_term", "long_term"})
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class SpotifyError(Exception): ...


class SpotifyHTTPError(SpotifyError): ...


class SpotifyAuthError(SpotifyError): ...


class SpotifyTrack(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    artists: list[str] = Field(default_factory=list)
    isrc: str | None = None
    link: str | None = None
    preview_url: str | None = None
    popularity: int | None = None
    duration_ms: int | None = None
    cover_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        artists = data.get("artists")
        if isinstance(artists, list):
            names = [a.get("name", "") for a in artists if isinstance(a, dict)]
            if names:
                out["artists"] = names
        ext_ids = data.get("external_ids")
        if isinstance(ext_ids, dict):
            out.setdefault("isrc", ext_ids.get("isrc"))
        ext_urls = data.get("external_urls")
        if isinstance(ext_urls, dict):
            out.setdefault("link", ext_urls.get("spotify"))
        album = data.get("album")
        if isinstance(album, dict):
            images = album.get("images")
            if isinstance(images, list) and images and isinstance(images[0], dict):
                out.setdefault("cover_url", images[0].get("url"))
        return out


class SpotifyClient:
    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        token_url: str = _TOKEN_URL,
        base_url: str = _API_BASE,
        timeout: float = _DEFAULT_TIMEOUT,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not client_id or not client_secret or not refresh_token:
            raise ValueError("client_id, client_secret, and refresh_token are required")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be > 0, got {max_concurrency}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if retry_backoff < 0:
            raise ValueError(f"retry_backoff must be >= 0, got {retry_backoff}")

        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._token_url = token_url
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = client
        self._owns_client = client is None
        self._access_token: str | None = None
        self._token_expiry = 0.0
        self._token_lock = asyncio.Lock()

    async def __aenter__(self) -> SpotifyClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url, timeout=self._timeout
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_track(self, track_id: str) -> SpotifyTrack | None:
        data = await self._get(f"/tracks/{track_id}")
        return SpotifyTrack.model_validate(data) if data is not None else None

    async def get_playlist_items(
        self, playlist_id: str, *, limit: int = 20
    ) -> list[SpotifyTrack]:
        data = await self._get(
            f"/playlists/{playlist_id}/tracks", params={"limit": _check_limit(limit)}
        )
        if data is None:
            return []
        return _tracks_from_items(data.get("items", []), nested=True)

    async def get_top_tracks(
        self, *, time_range: str = "medium_term", limit: int = 5
    ) -> list[SpotifyTrack]:
        if time_range not in _TIME_RANGES:
            raise ValueError(f"time_range must be one of {sorted(_TIME_RANGES)}")
        data = await self._get(
            "/me/top/tracks",
            params={"time_range": time_range, "limit": _check_limit(limit)},
        )
        if data is None:
            return []
        return _tracks_from_items(data.get("items", []), nested=False)

    async def get_recently_played(self, *, limit: int = 1) -> list[SpotifyTrack]:
        data = await self._get(
            "/me/player/recently-played", params={"limit": _check_limit(limit)}
        )
        if data is None:
            return []
        return _tracks_from_items(data.get("items", []), nested=True)

    async def _get(
        self, path: str, params: dict[str, object] | None = None
    ) -> dict[str, object] | None:
        if self._client is None:
            raise SpotifyError("client not started; use 'async with SpotifyClient()'")

        last: SpotifyError | None = None
        for attempt in range(self._max_retries + 1):
            token = await self._token()
            headers = {"Authorization": f"Bearer {token}"}
            try:
                async with self._semaphore:
                    response = await self._client.get(
                        path, params=params, headers=headers
                    )
            except httpx.HTTPError as exc:
                last = SpotifyHTTPError(f"transport error: {exc}")
                await self._sleep_backoff(attempt)
                continue

            if response.status_code == 404:
                return None
            if response.status_code == 401:
                self._token_expiry = 0.0
                last = SpotifyAuthError("unauthorized; access token rejected")
                continue
            if response.status_code in _RETRYABLE_STATUS:
                last = SpotifyHTTPError(f"http {response.status_code}")
                await self._sleep_backoff(attempt, response)
                continue
            if response.status_code >= 400:
                raise SpotifyHTTPError(
                    f"http {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            if not isinstance(data, dict):
                raise SpotifyError("expected Spotify response to be a JSON object")
            return data

        raise last or SpotifyHTTPError("request failed")

    async def _token(self) -> str:
        async with self._token_lock:
            if self._access_token is not None and monotonic() < self._token_expiry:
                return self._access_token
            return await self._refresh_access_token()

    async def _refresh_access_token(self) -> str:
        assert self._client is not None
        basic = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()
        ).decode()
        try:
            response = await self._client.post(
                self._token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                },
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        except httpx.HTTPError as exc:
            raise SpotifyAuthError(f"token request failed: {exc}") from exc

        if response.status_code != 200:
            raise SpotifyAuthError(
                f"token request returned {response.status_code}: {response.text[:200]}"
            )
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise SpotifyAuthError("token response missing access_token")
        expires_in = float(payload.get("expires_in", 3600))
        self._access_token = token
        self._token_expiry = monotonic() + expires_in - _TOKEN_EXPIRY_BUFFER
        new_refresh = payload.get("refresh_token")
        if new_refresh:
            self._refresh_token = new_refresh
        return token

    async def _sleep_backoff(
        self, attempt: int, response: httpx.Response | None = None
    ) -> None:
        if attempt >= self._max_retries:
            return
        retry_after = _parse_retry_after(response)
        if retry_after is not None:
            await asyncio.sleep(retry_after)
            return
        if self._retry_backoff > 0:
            await asyncio.sleep(self._retry_backoff * (2**attempt))


def _tracks_from_items(items: object, *, nested: bool) -> list[SpotifyTrack]:
    if not isinstance(items, list):
        raise SpotifyError("expected a list of items")
    tracks: list[SpotifyTrack] = []
    for item in items:
        raw = item.get("track") if nested else item
        if not isinstance(raw, dict):
            continue
        if not raw.get("id") or raw.get("type", "track") != "track":
            continue
        tracks.append(SpotifyTrack.model_validate(raw))
    return tracks


def _check_limit(limit: int) -> int:
    if not 1 <= limit <= _MAX_LIMIT:
        raise ValueError(f"limit must be in 1..{_MAX_LIMIT}, got {limit}")
    return limit


def _parse_retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
