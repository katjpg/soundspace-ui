import asyncio
from dataclasses import dataclass, field
from enum import Enum

from soundspace.services.recommend import RecommendService


class Readiness(str, Enum):
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ServiceState:
    model: str | None = None
    readiness: Readiness = Readiness.WARMING
    service: RecommendService | None = None
    error: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
