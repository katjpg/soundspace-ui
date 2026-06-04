from typing import Literal

from pydantic import Field

from soundspace.schemas.base import SchemaBase
from soundspace.services.recommend import (
    Recommendation,
    RecommendResult,
    SeedResolution,
)

RecommendMode = Literal["song", "playlist"]
TimeRange = Literal["short_term", "medium_term", "long_term"]


class RecommendRequest(SchemaBase):
    mode: RecommendMode = "song"
    text: str | None = None
    spotify_track: str | None = None
    spotify_playlist: str | None = None
    top: bool = False
    recent: bool = False
    time_range: TimeRange = "medium_term"
    n: int = Field(gt=0, le=50)
    model: str | None = None
    provider: str | None = None


class RecommendationItem(SchemaBase):
    song_id: str
    artist: str
    title: str
    region: str
    fit: float
    reason: str
    path: str | None = None


class SeedResolutionItem(SchemaBase):
    artist: str
    title: str
    resolved: bool
    deezer_id: int | None = None
    deezer_artist: str | None = None
    deezer_title: str | None = None
    score: float | None = None
    reason: str | None = None


class RecommendResponse(SchemaBase):
    recommendations: list[RecommendationItem]
    resolved_seeds: list[SeedResolutionItem]
    unresolved_seeds: list[SeedResolutionItem]
    candidate_count: int


def _to_item(rec: Recommendation) -> RecommendationItem:
    return RecommendationItem(
        song_id=rec.song_id,
        artist=rec.artist,
        title=rec.title,
        region=rec.region,
        fit=rec.fit,
        reason=rec.reason,
        path=str(rec.path) if rec.path is not None else None,
    )


def _to_seed(seed: SeedResolution) -> SeedResolutionItem:
    return SeedResolutionItem(
        artist=seed.artist,
        title=seed.title,
        resolved=seed.resolved,
        deezer_id=seed.deezer_id,
        deezer_artist=seed.deezer_artist,
        deezer_title=seed.deezer_title,
        score=seed.score,
        reason=seed.reason,
    )


def to_recommend_response(result: RecommendResult) -> RecommendResponse:
    return RecommendResponse(
        recommendations=[_to_item(r) for r in result.recommendations],
        resolved_seeds=[_to_seed(s) for s in result.resolved_seeds],
        unresolved_seeds=[_to_seed(s) for s in result.unresolved_seeds],
        candidate_count=result.candidate_count,
    )
