from fastapi import APIRouter, Depends

from soundspace.api.dependencies import get_state
from soundspace.api.state import ServiceState
from soundspace.schemas.base import SchemaBase

router = APIRouter(tags=["health"])


class HealthResponse(SchemaBase):
    status: str


class ReadyResponse(SchemaBase):
    readiness: str
    error: str | None = None


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="alive")


@router.get("/ready", response_model=ReadyResponse)
async def ready(state: ServiceState = Depends(get_state)) -> ReadyResponse:
    return ReadyResponse(readiness=state.readiness.value, error=state.error)
