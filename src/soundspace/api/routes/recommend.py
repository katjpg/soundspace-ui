import asyncio
import contextlib
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from soundspace.api.dependencies import get_lock, get_service
from soundspace.schemas.recommend import (
    RecommendRequest,
    RecommendResponse,
    to_recommend_response,
)
from soundspace.services.recommend import (
    RecommendError,
    RecommendResult,
    RecommendService,
)

router = APIRouter(tags=["recommend"])

_MAX_AUDIO_BYTES = 50 * 1024 * 1024


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    payload: str = Form(...),
    audio: UploadFile | None = File(default=None),
    service: RecommendService = Depends(get_service),
    lock: asyncio.Lock = Depends(get_lock),
) -> RecommendResponse:
    try:
        request = RecommendRequest.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid request: {exc}"
        ) from exc

    audio_bytes: bytes | None = None
    suffix = ".mp3"
    if audio is not None:
        audio_bytes = await audio.read()
        if len(audio_bytes) > _MAX_AUDIO_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "audio file too large"
            )
        suffix = Path(audio.filename or "seed").suffix or ".mp3"

    async with lock:
        try:
            result = await _dispatch(service, request, audio_bytes, suffix)
        except RecommendError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return to_recommend_response(result)


async def _dispatch(
    service: RecommendService,
    request: RecommendRequest,
    audio_bytes: bytes | None,
    suffix: str,
) -> RecommendResult:
    n = request.n
    if audio_bytes is not None:
        with _tempfile(audio_bytes, suffix=suffix) as path:
            return await service.from_audio(path, query=request.text, n=n)
    if request.spotify_track:
        return await service.from_spotify_track(
            request.spotify_track, query=request.text, n=n
        )
    if request.spotify_playlist:
        return await service.from_spotify_playlist(
            request.spotify_playlist, query=request.text, n=n
        )
    if request.top:
        return await service.from_top(
            time_range=request.time_range, query=request.text, n=n
        )
    if request.recent:
        return await service.from_recent(query=request.text, n=n)
    if request.text:
        return await service.from_text(request.text, n=n)
    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "no seed provided")


@contextlib.contextmanager
def _tempfile(data: bytes, *, suffix: str) -> Iterator[str]:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = handle.name
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)
