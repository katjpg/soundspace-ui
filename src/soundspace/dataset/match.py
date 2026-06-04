from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

_DEFAULT_ARTIST_THRESHOLD = 80.0
_DEFAULT_TITLE_THRESHOLD = 80.0
_ARTIST_WEIGHT = 0.5
_TITLE_WEIGHT = 0.5

_FEAT_PATTERN = re.compile(
    r"\s+[\(\[]?\s*(?:feat|ft|featuring|with)\b\.?\s.*", re.IGNORECASE
)
_BRACKET_PATTERN = re.compile(r"[\(\[].*?[\)\]]")
_PUNCT_PATTERN = re.compile(r"[^\w\s]")
_WS_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    track_id: int
    artist: str
    title: str
    rank: int | None = None
    duration: int | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    candidate: MatchCandidate
    artist_score: float
    title_score: float
    score: float


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = _FEAT_PATTERN.sub("", text)
    text = _BRACKET_PATTERN.sub(" ", text)
    text = _PUNCT_PATTERN.sub(" ", text)
    return _WS_PATTERN.sub(" ", text).strip()


def score_candidate(
    query_artist: str,
    query_title: str,
    candidate: MatchCandidate,
    *,
    artist_weight: float = _ARTIST_WEIGHT,
    title_weight: float = _TITLE_WEIGHT,
) -> MatchResult:
    artist_score = _ratio(query_artist, candidate.artist)
    title_score = _ratio(query_title, candidate.title)
    total = artist_weight + title_weight
    blended = (artist_score * artist_weight + title_score * title_weight) / total
    return MatchResult(
        candidate=candidate,
        artist_score=artist_score,
        title_score=title_score,
        score=blended,
    )


def best_match(
    query_artist: str,
    query_title: str,
    candidates: Sequence[MatchCandidate],
    *,
    artist_threshold: float = _DEFAULT_ARTIST_THRESHOLD,
    title_threshold: float = _DEFAULT_TITLE_THRESHOLD,
    artist_weight: float = _ARTIST_WEIGHT,
    title_weight: float = _TITLE_WEIGHT,
) -> MatchResult | None:
    qualified: list[MatchResult] = []
    for candidate in candidates:
        result = score_candidate(
            query_artist,
            query_title,
            candidate,
            artist_weight=artist_weight,
            title_weight=title_weight,
        )
        if (
            result.artist_score >= artist_threshold
            and result.title_score >= title_threshold
        ):
            qualified.append(result)

    if not qualified:
        return None
    return max(qualified, key=_rank_key)


def _ratio(query: str, candidate: str) -> float:
    q = normalize(query)
    c = normalize(candidate)
    if not q or not c:
        return 0.0
    return float(fuzz.token_set_ratio(q, c))


def _rank_key(result: MatchResult) -> tuple[float, int]:
    rank = result.candidate.rank if result.candidate.rank is not None else -1
    return (result.score, rank)
