import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from soundspace.client.deezer import DeezerClient, DeezerTrack
from soundspace.config.dataset import DatasetConfig
from soundspace.dataset.match import MatchCandidate, best_match
from soundspace.dataset.tracks import read_tracks

log = logging.getLogger(__name__)

_METADATA_SUFFIX = "_deezer_metadata.csv"
_SEARCH_LIMIT = 5
_MISS_SAMPLE = 10

_COLUMNS: tuple[str, ...] = (
    "song_id",
    "matched",
    "deezer_id",
    "deezer_artist",
    "deezer_title",
    "isrc",
    "link",
    "rank",
    "duration",
    "cover_small",
    "cover_medium",
    "cover_big",
    "cover_xl",
    "artist_picture_small",
    "artist_picture_medium",
    "artist_picture_big",
    "artist_picture_xl",
    "score",
    "artist_score",
    "title_score",
)


@dataclass(frozen=True, slots=True)
class _Query:
    song_id: str
    artist: str
    title: str


@dataclass(frozen=True, slots=True)
class _SearchOutcome:
    query: _Query
    candidates: list[DeezerTrack] | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FetchReport:
    path: Path
    total: int
    matched: int
    missed: int
    skipped: int
    miss_sample: list[_Query] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        attempted = self.total - self.skipped
        return self.matched / attempted if attempted else 0.0


def fetch_deezer(
    dataset: DatasetConfig,
    *,
    limit: int = _SEARCH_LIMIT,
    strict: bool = True,
    force: bool = False,
    progress: bool = True,
) -> FetchReport:
    table = read_tracks(dataset)
    _require_columns(table)

    out_path = dataset.processed_dir / f"{dataset.active}{_METADATA_SUFFIX}"
    existing = _load_existing(out_path) if not force else {}

    queries = _build_queries(table)
    pending = [q for q in queries if q.song_id not in existing]

    rows = list(existing.values())
    matched = sum(1 for r in rows if r.get("matched"))
    missed = sum(1 for r in rows if r.get("matched") is False)
    skipped = 0
    miss_sample: list[_Query] = []

    if pending:
        outcomes = asyncio.run(
            _search_all(pending, limit=limit, strict=strict, progress=progress)
        )
        for outcome in outcomes:
            if outcome.candidates is None:
                skipped += 1
                log.warning("skipped %s: %s", outcome.query.song_id, outcome.reason)
                continue
            row = _match_row(outcome.query, outcome.candidates)
            rows.append(row)
            if row["matched"]:
                matched += 1
            else:
                missed += 1
                if len(miss_sample) < _MISS_SAMPLE:
                    miss_sample.append(outcome.query)

    _write_rows(rows, out_path)

    return FetchReport(
        path=out_path,
        total=len(queries),
        matched=matched,
        missed=missed,
        skipped=skipped,
        miss_sample=miss_sample,
    )


async def _search_all(
    queries: list[_Query],
    *,
    limit: int,
    strict: bool,
    progress: bool,
) -> list[_SearchOutcome]:
    async with DeezerClient() as client:
        bar = tqdm(total=len(queries), desc="Fetching Deezer", disable=not progress)

        async def run(query: _Query) -> _SearchOutcome:
            outcome = await _search_one(client, query, limit=limit, strict=strict)
            bar.update(1)
            return outcome

        try:
            return await asyncio.gather(*(run(q) for q in queries))
        finally:
            bar.close()


async def _search_one(
    client: DeezerClient,
    query: _Query,
    *,
    limit: int,
    strict: bool,
) -> _SearchOutcome:
    try:
        candidates = await client.search_track(
            query.artist, query.title, strict=strict, limit=limit
        )
    except ValueError as exc:
        return _SearchOutcome(query=query, candidates=None, reason=str(exc))
    return _SearchOutcome(query=query, candidates=candidates)


def _match_row(query: _Query, candidates: list[DeezerTrack]) -> dict[str, object]:
    tracks_by_id = {track.id: track for track in candidates}
    pool = [
        MatchCandidate(
            track_id=track.id,
            artist=track.artist.name if track.artist else "",
            title=track.title,
            rank=track.rank,
            duration=track.duration,
        )
        for track in candidates
    ]
    result = best_match(query.artist, query.title, pool)
    if result is None:
        return _miss_row(query.song_id)

    track = tracks_by_id[result.candidate.track_id]
    album = track.album
    artist = track.artist
    return {
        "song_id": query.song_id,
        "matched": True,
        "deezer_id": track.id,
        "deezer_artist": artist.name if artist else None,
        "deezer_title": track.title,
        "isrc": track.isrc,
        "link": track.link,
        "rank": track.rank,
        "duration": track.duration,
        "cover_small": album.cover_small if album else None,
        "cover_medium": album.cover_medium if album else None,
        "cover_big": album.cover_big if album else None,
        "cover_xl": album.cover_xl if album else None,
        "artist_picture_small": artist.picture_small if artist else None,
        "artist_picture_medium": artist.picture_medium if artist else None,
        "artist_picture_big": artist.picture_big if artist else None,
        "artist_picture_xl": artist.picture_xl if artist else None,
        "score": round(result.score, 2),
        "artist_score": round(result.artist_score, 2),
        "title_score": round(result.title_score, 2),
    }


def _miss_row(song_id: str) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in _COLUMNS}
    row["song_id"] = song_id
    row["matched"] = False
    return row


def _build_queries(table: pd.DataFrame) -> list[_Query]:
    queries: list[_Query] = []
    for record in table.to_dict("records"):
        queries.append(
            _Query(
                song_id=str(record["song_id"]),
                artist=_text(record.get("artist")),
                title=_text(record.get("title")),
            )
        )
    return queries


def _load_existing(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"song_id": str})
    rows: dict[str, dict[str, object]] = {}
    for record in frame.to_dict("records"):
        record["matched"] = bool(record.get("matched"))
        rows[str(record["song_id"])] = record
    return rows


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(_COLUMNS))
    frame.to_csv(path, index=False)
    log.info("wrote %d rows -> %s", len(frame), path)


def _require_columns(table: pd.DataFrame) -> None:
    missing = {"song_id", "artist", "title"} - set(table.columns)
    if missing:
        raise ValueError(f"dataset table missing columns: {sorted(missing)}")


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()
