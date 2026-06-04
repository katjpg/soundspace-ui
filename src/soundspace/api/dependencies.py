import asyncio

from fastapi import HTTPException, Request, status

from soundspace.api.state import Readiness, ServiceState
from soundspace.services.recommend import RecommendService


def get_state(request: Request) -> ServiceState:
    return request.app.state.soundspace


def get_service(request: Request) -> RecommendService:
    state = get_state(request)
    if state.readiness is Readiness.WARMING:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "model still warming up"
        )
    if state.readiness is Readiness.FAILED or state.service is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"daemon warmup failed: {state.error}",
        )
    return state.service


def get_lock(request: Request) -> asyncio.Lock:
    return get_state(request).lock
