from __future__ import annotations

import asyncio
from collections import deque
from time import monotonic

import httpx
from pydantic import BaseModel, ConfigDict

_BASE_URL = "https://api.deezer.com"
_DEFAULT_RATE_LIMIT = 45
_DEFAULT_RATE_PERIOD = 5.0
_DEFAULT_MAX_CONCURRENCY = 8
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF = 1.0

_NOT_FOUND_CODE = 800
_QUOTA_CODES = (4, 700)
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class DeezerError(Exception):
    """Base error for Deezer client failures."""


class DeezerHTTPError(DeezerError):
    """Deezer returned an HTTP or transport failure."""


class DeezerQuotaError(DeezerError):
    """Deezer quota is exhausted."""


class DeezerArtist(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    name: str
    link: str | None = None
    picture_small: str | None = None
    picture_medium: str | None = None
    picture_big: str | None = None
    picture_xl: str | None = None


class DeezerAlbum(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    title: str
    link: str | None = None
    cover_small: str | None = None
    cover_medium: str | None = None
    cover_big: str | None = None
    cover_xl: str | None = None


class DeezerTrack(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    title: str
    duration: int | None = None
    isrc: str | None = None
    link: str | None = None
    preview: str | None = None
    rank: int | None = None
    artist: DeezerArtist | None = None
    album: DeezerAlbum | None = None


class _RateLimiter:
    def __init__(self, max_calls: int, period: float) -> None:
        if max_calls <= 0:
            raise ValueError(f"max_calls must be > 0, got {max_calls}")
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        self._max_calls = max_calls
        self._period = period
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = monotonic()
            self._evict(now)
            if len(self._calls) >= self._max_calls:
                wait = self._period - (now - self._calls[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                now = monotonic()
                self._evict(now)
            self._calls.append(monotonic())

    def _evict(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self._period:
            self._calls.popleft()


class DeezerClient:
    """Async client for Deezer catalogue lookups."""

    def __init__(
        self,
        *,
        base_url: str = _BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        rate_limit: int = _DEFAULT_RATE_LIMIT,
        rate_period: float = _DEFAULT_RATE_PERIOD,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be > 0, got {max_concurrency}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if retry_backoff < 0:
            raise ValueError(f"retry_backoff must be >= 0, got {retry_backoff}")

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._limiter = _RateLimiter(rate_limit, rate_period)
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> DeezerClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url, timeout=self._timeout
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search_track(
        self,
        artist: str,
        title: str,
        *,
        strict: bool = True,
        limit: int = 5,
    ) -> list[DeezerTrack]:
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")

        params: dict[str, object] = {"q": _track_query(artist, title), "limit": limit}
        if strict:
            params["strict"] = "on"

        data = await self._get("/search", params)
        if data is None:
            return []

        rows = data.get("data", [])
        if not isinstance(rows, list):
            raise DeezerError("search response missing data list")
        return [DeezerTrack.model_validate(row) for row in rows]

    async def get_track(self, track_id: int | str) -> DeezerTrack | None:
        data = await self._get(f"/track/{track_id}")
        return DeezerTrack.model_validate(data) if data is not None else None

    async def get_artist(self, artist_id: int | str) -> DeezerArtist | None:
        data = await self._get(f"/artist/{artist_id}")
        return DeezerArtist.model_validate(data) if data is not None else None

    async def get_preview_url(self, track_id: int | str) -> str | None:
        track = await self.get_track(track_id)
        return track.preview if track is not None else None

    async def _get(
        self,
        path: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if self._client is None:
            raise DeezerError("client not started; use 'async with DeezerClient()'")

        last: DeezerError | None = None
        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire()
            try:
                async with self._semaphore:
                    response = await self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                last = DeezerHTTPError(f"transport error: {exc}")
                await self._sleep_backoff(attempt)
                continue

            if response.status_code in _RETRYABLE_STATUS:
                last = DeezerHTTPError(f"http {response.status_code}")
                await self._sleep_backoff(attempt)
                continue
            if response.status_code >= 400:
                raise DeezerHTTPError(
                    f"http {response.status_code}: {response.text[:200]}"
                )

            data = response.json()
            if isinstance(data, dict) and "error" in data:
                kind = _classify_deezer_error(data["error"])
                if kind == "not_found":
                    return None
                if kind == "quota":
                    last = DeezerQuotaError(str(data["error"]))
                    await self._sleep_backoff(attempt)
                    continue
                raise DeezerError(str(data["error"]))
            if not isinstance(data, dict):
                raise DeezerError("expected Deezer response to be a JSON object")
            return data

        raise last or DeezerHTTPError("request failed")

    async def _sleep_backoff(self, attempt: int) -> None:
        if attempt < self._max_retries and self._retry_backoff > 0:
            await asyncio.sleep(self._retry_backoff * (2**attempt))


def _track_query(artist: str, title: str) -> str:
    artist = artist.replace('"', "").strip()
    title = title.replace('"', "").strip()
    if not artist:
        raise ValueError("artist must not be empty")
    if not title:
        raise ValueError("title must not be empty")
    return f'artist:"{artist}" track:"{title}"'


def _classify_deezer_error(error: object) -> str:
    if not isinstance(error, dict):
        return "error"
    error_type = str(error.get("type", ""))
    code = error.get("code")
    if error_type == "DataException" or code == _NOT_FOUND_CODE:
        return "not_found"
    if error_type == "QuotaException" or code in _QUOTA_CODES:
        return "quota"
    return "error"
