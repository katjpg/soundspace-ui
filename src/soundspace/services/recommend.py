import asyncio
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from soundspace.client.deezer import DeezerClient
from soundspace.client.spotify import SpotifyClient, SpotifyTrack
from soundspace.config.dataset import DatasetConfig
from soundspace.config.llm import RerankConfig
from soundspace.dataset.match import MatchCandidate, best_match
from soundspace.dataset.tracks import read_tracks, resolve_audio_paths
from soundspace.llm.agents.rank import RankAgent, RankedSong
from soundspace.llm.providers.base import LLMProvider
from soundspace.space.embed.base import Embedder
from soundspace.space.search.index import SearchIndex
from soundspace.space.search.query import (
    embed_audio_query,
    embed_text_query,
    preprocess_query,
    retrieve_top_k,
)

_EMBEDDINGS_SUBDIR = "embeddings"
_PREVIEW_TIMEOUT = 15.0
_PLAYLIST_SEED_MAX = 20
_PLAYLIST_PER_SEED_K = 3
_PLAYLIST_FIT_FLOOR = 0.45
_SPOTIFY_TRACK_RE = re.compile(
    r"(?:open\.spotify\.com/track/|spotify:track:)([A-Za-z0-9]+)"
)
_SPOTIFY_PLAYLIST_RE = re.compile(
    r"(?:open\.spotify\.com/playlist/|spotify:playlist:)([A-Za-z0-9]+)"
)
_EVIDENCE_TAGS = ("style", "mood", "theme")

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


class RecommendError(Exception): ...


@dataclass(frozen=True, slots=True)
class SeedResolution:
    artist: str
    title: str
    resolved: bool
    deezer_id: int | None = None
    deezer_artist: str | None = None
    deezer_title: str | None = None
    score: float | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SeedVector:
    vector: np.ndarray
    merge_id: str | None


@dataclass(frozen=True, slots=True)
class Recommendation:
    song_id: str
    artist: str
    title: str
    region: str
    fit: float
    reason: str
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class RecommendResult:
    recommendations: list[Recommendation]
    resolved_seeds: list[SeedResolution] = field(default_factory=list)
    unresolved_seeds: list[SeedResolution] = field(default_factory=list)
    candidate_count: int = 0


def parse_spotify_track_id(value: str) -> str:
    match = _SPOTIFY_TRACK_RE.search(value)
    return match.group(1) if match else value.strip()


def parse_spotify_playlist_id(value: str) -> str:
    match = _SPOTIFY_PLAYLIST_RE.search(value)
    return match.group(1) if match else value.strip()


class RecommendService:
    def __init__(
        self,
        dataset: DatasetConfig,
        *,
        model: str,
        embedder: Embedder,
        provider: LLMProvider,
        config: RerankConfig,
        spotify: SpotifyClient | None = None,
        deezer: DeezerClient | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self._dataset = dataset
        self._model = model
        self._embedder = embedder
        self._agent = RankAgent(provider=provider)
        self._config = config
        self._spotify = spotify
        self._deezer = deezer
        self._progress = progress or _noop
        self._progress(f"Loading index '{model}'")
        self._index = SearchIndex.load(self._index_path())
        table = read_tracks(dataset)
        self._meta = table.set_index("song_id")
        self._paths = resolve_audio_paths(table, dataset.audio_dir)
        self._labels = self._load_labels()
        self._progress(f"Index ready: {self._index.n_tracks} tracks")

    async def from_text(self, query: str, *, n: int) -> RecommendResult:
        self._progress(f"Embedding text query: {query!r}")
        vector = preprocess_query(embed_text_query(query, self._embedder), self._index)
        hits = self._retrieve_single(vector, k=n)
        self._progress(f"Retrieved {len(hits)} candidates; ranking with LLM")
        ranked = await self._rank_single(query, hits)
        return RecommendResult(
            recommendations=self._to_recommendations(ranked, n=n),
            candidate_count=len(hits),
        )

    async def from_audio(
        self, path: str | Path, *, query: str | None, n: int
    ) -> RecommendResult:
        self._progress(f"Embedding audio file: {path}")
        vector = preprocess_query(embed_audio_query(path, self._embedder), self._index)
        hits = self._retrieve_single(vector, k=n)
        if query:
            self._progress(f"Retrieved {len(hits)} candidates; ranking with LLM")
            ranked = await self._rank_single(query, hits)
            return RecommendResult(
                recommendations=self._to_recommendations(ranked, n=n),
                candidate_count=len(hits),
            )
        self._progress(f"Retrieved {len(hits)} candidates by similarity (no LLM)")
        return RecommendResult(
            recommendations=self._from_similarity(hits, n=n),
            candidate_count=len(hits),
        )

    async def from_spotify_track(
        self, ref: str, *, query: str | None, n: int
    ) -> RecommendResult:
        track_id = parse_spotify_track_id(ref)
        self._progress(f"Fetching Spotify track {track_id}")
        track = await self._require_spotify().get_track(track_id)
        if track is None:
            raise RecommendError("Spotify track not found")
        self._progress(f"Spotify: {_track_label(track)}")
        resolved, unresolved = await self._resolve_seeds([track])
        if not resolved:
            raise RecommendError("seed track could not be matched to playable audio")
        seed_vectors = await self._embed_seeds(resolved)
        if not seed_vectors:
            raise RecommendError("seed preview could not be embedded")

        seed = seed_vectors[0]
        exclude = [seed.merge_id] if seed.merge_id else []
        hits = self._retrieve_single(seed.vector, k=n, exclude=exclude)
        if query:
            self._progress(f"Retrieved {len(hits)} candidates; ranking with LLM")
            ranked = await self._rank_single(query, hits)
            recs = self._to_recommendations(ranked, n=n)
        else:
            self._progress(f"Retrieved {len(hits)} candidates by similarity (no LLM)")
            recs = self._from_similarity(hits, n=n)
        return RecommendResult(
            recommendations=recs,
            resolved_seeds=resolved,
            unresolved_seeds=unresolved,
            candidate_count=len(hits),
        )

    async def from_spotify_playlist(
        self, ref: str, *, query: str | None, n: int
    ) -> RecommendResult:
        playlist_id = parse_spotify_playlist_id(ref)
        self._progress(f"Fetching Spotify playlist {playlist_id}")
        tracks = await self._require_spotify().get_playlist_items(
            playlist_id, limit=_PLAYLIST_SEED_MAX
        )
        if not tracks:
            raise RecommendError("Spotify playlist had no usable tracks")
        return await self._recommend_playlist(
            tracks[:_PLAYLIST_SEED_MAX], query=query, n=n
        )

    async def from_top(
        self, *, time_range: str, query: str | None, n: int
    ) -> RecommendResult:
        self._progress(f"Fetching your top tracks ({time_range})")
        tracks = await self._require_spotify().get_top_tracks(
            time_range=time_range, limit=_PLAYLIST_SEED_MAX
        )
        if not tracks:
            raise RecommendError("Spotify returned no top tracks")
        return await self._recommend_playlist(
            tracks[:_PLAYLIST_SEED_MAX], query=query, n=n
        )

    async def from_recent(self, *, query: str | None, n: int) -> RecommendResult:
        self._progress("Fetching recently played tracks")
        tracks = await self._require_spotify().get_recently_played(
            limit=_PLAYLIST_SEED_MAX
        )
        if not tracks:
            raise RecommendError("Spotify returned no recently played tracks")
        return await self._recommend_playlist(
            tracks[:_PLAYLIST_SEED_MAX], query=query, n=n
        )

    async def _recommend_playlist(
        self, tracks: Sequence[SpotifyTrack], *, query: str | None, n: int
    ) -> RecommendResult:
        n = min(n, _PLAYLIST_SEED_MAX)
        self._progress(f"Using {len(tracks)} playlist seeds")
        resolved, unresolved = await self._resolve_seeds(tracks)
        if not resolved:
            raise RecommendError("no playlist seeds could be matched to playable audio")
        seed_vectors = await self._embed_seeds(resolved)
        if not seed_vectors:
            raise RecommendError("no playlist seed previews could be embedded")

        self._progress(
            f"Retrieving top {_PLAYLIST_PER_SEED_K} per seed from {len(seed_vectors)} seeds"
        )
        pool = self._retrieve_pool(seed_vectors)
        candidates = self._build_evidence(pool)
        if not candidates:
            raise RecommendError("retrieval produced no candidates")

        self._progress(f"Pooled {len(candidates)} unique candidates; ranking with LLM")
        intent = query or _describe_seeds(tracks)
        seed_summaries = [{"artist": s.artist, "title": s.title} for s in resolved]
        ranked = await self._agent.rank_playlist(intent, candidates, seed_summaries)
        kept = [song for song in ranked if song.fit >= _PLAYLIST_FIT_FLOOR]
        dropped = len(ranked) - len(kept)
        if dropped:
            self._progress(
                f"Dropped {dropped} candidates below fit floor ({_PLAYLIST_FIT_FLOOR})"
            )
        return RecommendResult(
            recommendations=self._to_recommendations(kept, n=n),
            resolved_seeds=resolved,
            unresolved_seeds=unresolved,
            candidate_count=len(candidates),
        )

    def _retrieve_single(
        self, vector: np.ndarray, *, k: int, exclude: Sequence[str] = ()
    ) -> list[tuple[str, float]]:
        return retrieve_top_k(vector, self._index, k=k, exclude=exclude)

    def _retrieve_pool(self, seeds: Sequence[SeedVector]) -> dict[str, float]:
        support: dict[str, float] = {}
        for seed in seeds:
            exclude = [seed.merge_id] if seed.merge_id else []
            for song_id, score in retrieve_top_k(
                seed.vector, self._index, k=_PLAYLIST_PER_SEED_K, exclude=exclude
            ):
                support[song_id] = max(support.get(song_id, 0.0), float(score))
        return support

    async def _rank_single(
        self, query: str, hits: list[tuple[str, float]]
    ) -> list[RankedSong]:
        candidates = self._build_evidence(dict(hits))
        if not candidates:
            raise RecommendError("retrieval produced no candidates")
        return await self._agent.rank(query, candidates)

    def _from_similarity(
        self, hits: list[tuple[str, float]], *, n: int
    ) -> list[Recommendation]:
        out: list[Recommendation] = []
        for song_id, score in hits[:n]:
            out.append(
                self._recommendation(song_id, fit=round(float(score), 4), reason="")
            )
        return out

    def _build_evidence(self, pool: dict[str, float]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for song_id, similarity in pool.items():
            if song_id not in self._meta.index:
                continue
            meta = self._meta.loc[song_id]
            row: dict[str, object] = {
                "id": song_id,
                "title": _cell(meta.get("title")),
                "artist": _cell(meta.get("artist")),
                "valence": _num(meta.get("valence")),
                "arousal": _num(meta.get("arousal")),
                "similarity": round(similarity, 4),
            }
            for tag in _EVIDENCE_TAGS:
                row[tag] = _split_tags(meta.get(tag))
            rows.append(row)
        return rows

    def _to_recommendations(
        self, ranked: Sequence[RankedSong], *, n: int
    ) -> list[Recommendation]:
        return [
            self._recommendation(song.id, fit=song.fit, reason=song.reason)
            for song in ranked[:n]
        ]

    def _recommendation(
        self, song_id: str, *, fit: float, reason: str
    ) -> Recommendation:
        artist = title = ""
        if song_id in self._meta.index:
            meta = self._meta.loc[song_id]
            artist = _cell(meta.get("artist"))
            title = _cell(meta.get("title"))
        return Recommendation(
            song_id=song_id,
            artist=artist,
            title=title,
            region=self._labels.get(song_id, ""),
            fit=fit,
            reason=reason,
            path=self._paths.get(song_id),
        )

    async def _resolve_seeds(
        self, tracks: Sequence[SpotifyTrack]
    ) -> tuple[list[SeedResolution], list[SeedResolution]]:
        deezer = self._require_deezer()
        self._progress(f"Matching {len(tracks)} seed(s) on Deezer")
        results = await asyncio.gather(*(self._resolve_one(deezer, t) for t in tracks))
        resolved = [r for r in results if r.resolved]
        unresolved = [r for r in results if not r.resolved]
        for r in resolved:
            self._progress(
                f"  matched '{r.artist} - {r.title}' -> Deezer "
                f"'{r.deezer_artist} - {r.deezer_title}' (score {r.score:.0f})"
            )
        for r in unresolved:
            self._progress(f"  unmatched '{r.artist} - {r.title}': {r.reason}")
        return resolved, unresolved

    async def _resolve_one(
        self, deezer: DeezerClient, track: SpotifyTrack
    ) -> SeedResolution:
        artist = track.artists[0] if track.artists else ""
        title = track.name
        if not artist or not title:
            return SeedResolution(
                artist, title, resolved=False, reason="missing artist/title"
            )
        try:
            candidates = await deezer.search_track(artist, title)
        except ValueError as exc:
            return SeedResolution(artist, title, resolved=False, reason=str(exc))
        pool = [
            MatchCandidate(
                track_id=c.id,
                artist=c.artist.name if c.artist else "",
                title=c.title,
                rank=c.rank,
                duration=c.duration,
            )
            for c in candidates
        ]
        match = best_match(artist, title, pool)
        if match is None:
            return SeedResolution(
                artist, title, resolved=False, reason="no Deezer match"
            )
        return SeedResolution(
            artist,
            title,
            resolved=True,
            deezer_id=match.candidate.track_id,
            deezer_artist=match.candidate.artist,
            deezer_title=match.candidate.title,
            score=match.score,
        )

    async def _embed_seeds(self, seeds: Sequence[SeedResolution]) -> list[SeedVector]:
        deezer = self._require_deezer()
        vectors: list[SeedVector] = []
        for seed in seeds:
            if seed.deezer_id is None:
                continue
            url = await deezer.get_preview_url(seed.deezer_id)
            if not url:
                self._progress(
                    f"  no preview for '{seed.artist} - {seed.title}'; skipping"
                )
                continue
            self._progress(
                f"  downloading + embedding preview for '{seed.artist} - {seed.title}'"
            )
            audio = await self._download_preview(url)
            if audio is None:
                self._progress(
                    f"  preview download failed for '{seed.artist} - {seed.title}'"
                )
                continue
            vector = self._embed_preview_bytes(audio)
            if vector is None:
                self._progress(
                    f"  preview embed failed for '{seed.artist} - {seed.title}'"
                )
                continue
            merge_id = self._corpus_id(seed)
            vectors.append(SeedVector(vector=vector, merge_id=merge_id))
        self._progress(f"Embedded {len(vectors)} seed preview(s)")
        return vectors

    def _corpus_id(self, seed: SeedResolution) -> str | None:
        if not seed.artist or not seed.title:
            return None
        title = str(seed.title).strip().casefold()
        artist = str(seed.artist).strip().casefold()
        for song_id, row in zip(self._meta.index, self._meta.itertuples()):
            if (
                str(getattr(row, "title", "")).strip().casefold() == title
                and str(getattr(row, "artist", "")).strip().casefold() == artist
            ):
                return str(song_id)
        return None

    async def _download_preview(self, url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=_PREVIEW_TIMEOUT) as client:
                response = await client.get(url)
        except httpx.HTTPError:
            return None
        if response.status_code != 200 or not response.content:
            return None
        return response.content

    def _embed_preview_bytes(self, audio: bytes) -> np.ndarray | None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            handle.write(audio)
            tmp_path = handle.name
        try:
            raw = embed_audio_query(tmp_path, self._embedder)
            return preprocess_query(raw, self._index)
        except Exception:
            return None
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _index_path(self) -> Path:
        return self._dataset.processed_dir / _EMBEDDINGS_SUBDIR / f"{self._model}.npz"

    def _load_labels(self) -> dict[str, str]:
        path = self._dataset.artifacts_dir / f"{self._model}_clusters.csv"
        if not path.exists():
            return {}
        frame = pd.read_csv(path, dtype={"song_id": str})
        return dict(zip(frame["song_id"], frame["label"]))

    def _require_spotify(self) -> SpotifyClient:
        if self._spotify is None:
            raise RecommendError("Spotify is not configured; set SPOTIFY_* credentials")
        return self._spotify

    def _require_deezer(self) -> DeezerClient:
        if self._deezer is None:
            raise RecommendError("Deezer client is not available")
        return self._deezer


def _track_label(track: SpotifyTrack) -> str:
    artist = track.artists[0] if track.artists else "?"
    return f"{artist} - {track.name}"


def _describe_seeds(tracks: Sequence[SpotifyTrack]) -> str:
    names = [_track_label(t) for t in tracks[:5]]
    return "tracks in the spirit of: " + "; ".join(names)


def _split_tags(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _num(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
