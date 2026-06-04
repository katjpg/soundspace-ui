import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from soundspace.api.router import router
from soundspace.api.state import Readiness, ServiceState
from soundspace.client.deezer import DeezerClient
from soundspace.client.spotify import SpotifyClient
from soundspace.config.settings import get_settings
from soundspace.services import runtime
from soundspace.services.recommend import RecommendService
from soundspace.space.embed.base import load_embedder

log = logging.getLogger(__name__)


def create_app(*, model: str | None = None) -> FastAPI:
    app = FastAPI(title="SoundSpace", lifespan=_lifespan)
    app.state.soundspace = ServiceState(model=model)
    app.include_router(router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    state: ServiceState = app.state.soundspace
    async with AsyncExitStack() as stack:
        deezer = await stack.enter_async_context(DeezerClient())
        spotify_factory = runtime.spotify_client(get_settings())
        spotify = (
            await stack.enter_async_context(spotify_factory)
            if spotify_factory is not None
            else None
        )
        warmup = asyncio.create_task(_warmup(state, deezer=deezer, spotify=spotify))
        try:
            yield
        finally:
            warmup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warmup


async def _warmup(
    state: ServiceState,
    *,
    deezer: DeezerClient,
    spotify: SpotifyClient | None,
) -> None:
    try:
        service = await asyncio.to_thread(
            _build_service, state.model, deezer=deezer, spotify=spotify
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("daemon warmup failed")
        state.error = str(exc)
        state.readiness = Readiness.FAILED
        return
    state.service = service
    state.readiness = Readiness.READY
    log.info("daemon ready")


def _build_service(
    model: str | None,
    *,
    deezer: DeezerClient,
    spotify: SpotifyClient | None,
) -> RecommendService:
    settings = get_settings()
    dataset = runtime.dataset_config()
    config = runtime.embedding_config(model)
    provider = runtime.provider(settings, None)
    embedder = load_embedder(config)
    _ = embedder.dim
    return RecommendService(
        dataset,
        model=config.name,
        embedder=embedder,
        provider=provider,
        config=settings.llm.rerank,
        spotify=spotify,
        deezer=deezer,
        progress=None,
    )
