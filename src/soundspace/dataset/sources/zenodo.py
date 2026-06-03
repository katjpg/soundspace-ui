from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from tqdm import tqdm

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120.0
_DEFAULT_CHUNK_SIZE = 1024 * 1024
_DEFAULT_MAX_RETRIES = 4
_DEFAULT_BACKOFF = 1.5
_DEFAULT_BACKOFF_CAP = 30.0
_USER_AGENT = "soundspace/0.1 (ZenodoClient)"

_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class ZenodoError(Exception):
    """Base error for Zenodo failures."""


class ZenodoHTTPError(ZenodoError):
    """Zenodo returned a failed response."""


class ZenodoFileNotFoundError(ZenodoError):
    """A record does not contain the requested file."""


class ZenodoChecksumError(ZenodoError):
    """A downloaded file does not match its checksum."""


@dataclass(frozen=True, slots=True)
class ZenodoConfig:
    base_url: str = "https://zenodo.org"
    token: str | None = None
    timeout: float = _DEFAULT_TIMEOUT
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    max_retries: int = _DEFAULT_MAX_RETRIES
    backoff: float = _DEFAULT_BACKOFF
    backoff_cap: float = _DEFAULT_BACKOFF_CAP
    user_agent: str = _USER_AGENT
    show_progress: bool = True

    @property
    def api_base(self) -> str:
        return f"{self.base_url.rstrip('/')}/api"


@dataclass(frozen=True, slots=True)
class ZenodoFile:
    key: str
    size: int | None
    checksum: str | None
    links: dict[str, str]

    @property
    def md5(self) -> str | None:
        if self.checksum is None:
            return None
        return self.checksum.strip().lower().removeprefix("md5:")

    @property
    def download_url(self) -> str | None:
        return (
            self.links.get("self")
            or self.links.get("content")
            or self.links.get("download")
        )


@dataclass(frozen=True, slots=True)
class ZenodoRecord:
    id: int
    conceptrecid: str | None
    metadata: dict[str, Any]
    files: list[ZenodoFile]


class ZenodoClient:
    def __init__(self, config: ZenodoConfig | None = None) -> None:
        self.config = config or ZenodoConfig()
        self._client: httpx.Client | None = None

    def __enter__(self) -> ZenodoClient:
        headers = {"User-Agent": self.config.user_agent}
        if self.config.token is not None:
            headers["Authorization"] = f"Bearer {self.config.token}"
        self._client = httpx.Client(
            base_url=self.config.api_base,
            headers=headers,
            timeout=self.config.timeout,
            follow_redirects=True,
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __repr__(self) -> str:
        status = "connected" if self._client is not None else "uninitialised"
        return f"ZenodoClient(base_url={self.config.base_url!r}, status={status!r})"

    @property
    def _c(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("ZenodoClient must be used as a context manager.")
        return self._client

    def get_record(self, record_id: int | str) -> ZenodoRecord:
        data = self._request(
            "GET",
            f"/records/{record_id}",
            headers={"Accept": "application/json"},
        ).json()
        files = [
            ZenodoFile(
                key=entry["key"],
                size=entry.get("size"),
                checksum=entry.get("checksum"),
                links=entry.get("links", {}),
            )
            for entry in data.get("files", [])
        ]
        return ZenodoRecord(
            id=int(data["id"]),
            conceptrecid=data.get("conceptrecid"),
            metadata=data.get("metadata", {}),
            files=files,
        )

    def list_files(self, record_id: int | str) -> list[ZenodoFile]:
        return self.get_record(record_id).files

    def find_file(self, record_id: int | str, filename: str) -> ZenodoFile:
        for entry in self.get_record(record_id).files:
            if entry.key == filename:
                return entry
        raise ZenodoFileNotFoundError(
            f"{filename!r} not found in zenodo record {record_id}"
        )

    def download_record_file(
        self,
        record_id: int | str,
        filename: str,
        dest: str | Path,
        *,
        expected_md5: str | None = None,
        force: bool = False,
    ) -> Path:
        entry = self.find_file(record_id, filename)
        url = entry.download_url
        if url is None:
            base_url = self.config.base_url.rstrip("/")
            url = f"{base_url}/records/{record_id}/files/{quote(filename)}?download=1"
        return self.download_url(
            url,
            dest,
            expected_md5=expected_md5 or entry.md5,
            force=force,
        )

    def download_url(
        self,
        url: str,
        dest: str | Path,
        *,
        expected_md5: str | None = None,
        force: bool = False,
    ) -> Path:
        dest = Path(dest)
        if dest.exists() and not force:
            if expected_md5 is None:
                log.info("reusing %s", dest)
                return dest
            if self.verify_md5(dest, expected_md5):
                log.info("reusing %s", dest)
                return dest
            log.warning("%s failed checksum, downloading again", dest)

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        log.info("downloading %s", dest.name)
        actual_md5 = self._stream_to_file(url, tmp, label=dest.name)

        if expected_md5 is not None:
            expected = _normalize_md5(expected_md5)
            if actual_md5 != expected:
                tmp.unlink(missing_ok=True)
                raise ZenodoChecksumError(
                    f"checksum mismatch for {dest.name}: "
                    f"expected {expected}, got {actual_md5}"
                )

        tmp.replace(dest)
        log.info("saved %s", dest)
        return dest

    def verify_md5(self, path: str | Path, expected_md5: str) -> bool:
        expected = _normalize_md5(expected_md5)
        return _compute_md5(Path(path), self.config.chunk_size) == expected

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._c.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                last = exc
                if attempt >= self.config.max_retries:
                    break
                log.warning(
                    "zenodo %s %s failed: %s; retry %d",
                    method,
                    url,
                    exc,
                    attempt,
                )
                self._sleep_backoff(attempt)
                continue
            if response.is_error:
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and attempt < self.config.max_retries
                ):
                    log.warning(
                        "zenodo %s %s returned %d; retry %d",
                        method,
                        url,
                        response.status_code,
                        attempt,
                    )
                    self._sleep_backoff(attempt, response)
                    continue
                raise ZenodoHTTPError(
                    f"{response.status_code} {response.reason_phrase}: "
                    f"{response.text[:500]}"
                )
            return response
        raise ZenodoHTTPError(
            f"request to {url} failed after {self.config.max_retries} attempts"
        ) from last

    def _stream_to_file(self, url: str, tmp: Path, *, label: str) -> str:
        last: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            tmp.unlink(missing_ok=True)
            try:
                with self._c.stream("GET", url) as response:
                    if response.is_error:
                        response.read()
                        if (
                            response.status_code in _RETRYABLE_STATUS
                            and attempt < self.config.max_retries
                        ):
                            log.warning(
                                "zenodo download %s returned %d; retry %d",
                                url,
                                response.status_code,
                                attempt,
                            )
                            self._sleep_backoff(attempt, response)
                            continue
                        raise ZenodoHTTPError(
                            f"{response.status_code} {response.reason_phrase}: "
                            f"{response.text[:500]}"
                        )
                    total = _parse_content_length(response)
                    return self._write_response(response, tmp, label=label, total=total)
            except httpx.TransportError as exc:
                last = exc
                if attempt >= self.config.max_retries:
                    break
                log.warning(
                    "zenodo download %s interrupted: %s; retry %d",
                    url,
                    exc,
                    attempt,
                )
                self._sleep_backoff(attempt)
        tmp.unlink(missing_ok=True)
        raise ZenodoHTTPError(
            f"download of {url} failed after {self.config.max_retries} attempts"
        ) from last

    def _write_response(
        self,
        response: httpx.Response,
        tmp: Path,
        *,
        label: str,
        total: int | None,
    ) -> str:
        hasher = hashlib.md5()
        progress = _progress_bar(label, total) if self.config.show_progress else None
        try:
            with tmp.open("wb") as file:
                for chunk in response.iter_bytes(self.config.chunk_size):
                    if not chunk:
                        continue
                    file.write(chunk)
                    hasher.update(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
        finally:
            if progress is not None:
                progress.close()
        return hasher.hexdigest()

    def _sleep_backoff(
        self,
        attempt: int,
        response: httpx.Response | None = None,
    ) -> None:
        retry_after = _parse_retry_after(response)
        if retry_after is not None:
            time.sleep(min(retry_after, self.config.backoff_cap))
            return
        delay = min(
            self.config.backoff_cap,
            self.config.backoff * (2 ** (attempt - 1)),
        )
        time.sleep(delay)


def _compute_md5(path: Path, chunk_size: int) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_md5(value: str) -> str:
    return value.strip().lower().removeprefix("md5:")


def _parse_content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


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


def _progress_bar(label: str, total: int | None) -> tqdm:
    return tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=label,
    )
