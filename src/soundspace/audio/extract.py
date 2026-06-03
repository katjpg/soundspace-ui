import logging
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from soundspace.audio.feature import (
    ALL_FEATURE_GROUPS,
    SAMPLE_RATE,
    FeatureGroup,
    RhythmFeatures,
    SpectralFeatures,
    TonalFeatures,
    compute_rhythm,
    compute_spectral,
    compute_tonal,
    load_audio,
)
from soundspace.config.dataset import DatasetConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrackFeatures:
    song_id: str
    rhythm: RhythmFeatures | None = None
    spectral: SpectralFeatures | None = None
    tonal: TonalFeatures | None = None

    def to_row(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {"song_id": self.song_id}
        if self.rhythm is not None:
            row.update(asdict(self.rhythm))
        if self.spectral is not None:
            row.update(asdict(self.spectral))
        if self.tonal is not None:
            row.update(asdict(self.tonal))
        return row


@dataclass(frozen=True, slots=True)
class FeatureFailure:
    song_id: str
    error: str


@dataclass(frozen=True, slots=True)
class FeatureSet:
    tracks: list[TrackFeatures]
    groups: tuple[FeatureGroup, ...]
    failures: list[FeatureFailure] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self) -> Iterator[TrackFeatures]:
        return iter(self.tracks)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([track.to_row() for track in self.tracks])


def extract(
    cfg: DatasetConfig,
    *,
    feature_groups: Sequence[FeatureGroup] = ALL_FEATURE_GROUPS,
    num_workers: int | None = None,
    song_ids: Sequence[str] | None = None,
    progress: bool = True,
) -> FeatureSet:
    groups = _check_feature_groups(feature_groups)
    workers = _resolve_workers(num_workers)

    table = pd.read_csv(cfg.processed_dir / f"{cfg.active}.csv", dtype={"song_id": str})
    audio_paths = _resolve_paths(table, cfg.audio_dir)
    audio_paths = _select_paths(audio_paths, song_ids)

    order = {song_id: index for index, song_id in enumerate(audio_paths)}
    tasks = [(song_id, str(path), groups) for song_id, path in audio_paths.items()]

    features: list[TrackFeatures] = []
    failures: list[FeatureFailure] = []

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_extract_track, task): task[0] for task in tasks}
        completed = as_completed(futures)
        if progress:
            completed = tqdm(completed, total=len(futures), desc="Extracting features")
        for future in completed:
            song_id = futures[future]
            try:
                features.append(future.result())
            except Exception as exc:
                log.warning("failed to extract features for %s: %s", song_id, exc)
                failures.append(FeatureFailure(song_id=song_id, error=str(exc)))

    features.sort(key=lambda track: order[track.song_id])

    return FeatureSet(tracks=features, groups=groups, failures=failures)


def _resolve_paths(table: pd.DataFrame, audio_dir: Path) -> dict[str, Path]:
    _check_columns(table, {"song_id"})

    has_audio_path = "audio_path" in table.columns
    has_quadrant = "quadrant" in table.columns

    paths: dict[str, Path] = {}
    for row in table.to_dict("records"):
        song_id = str(row["song_id"])
        if has_audio_path and isinstance(row["audio_path"], str):
            path = Path(row["audio_path"])
            paths[song_id] = path if path.is_absolute() else audio_dir / path
        elif has_quadrant:
            paths[song_id] = audio_dir / str(row["quadrant"]) / f"{song_id}.mp3"
        else:
            paths[song_id] = audio_dir / f"{song_id}.mp3"
    return paths


def _select_paths(
    paths: dict[str, Path],
    song_ids: Sequence[str] | None,
) -> dict[str, Path]:
    if song_ids is None:
        return paths
    selected = {str(song_id) for song_id in song_ids}
    return {song_id: path for song_id, path in paths.items() if song_id in selected}


def _extract_track(
    task: tuple[str, str, tuple[FeatureGroup, ...]],
) -> TrackFeatures:
    song_id, audio_path, groups = task
    audio = load_audio(Path(audio_path))

    rhythm = (
        compute_rhythm(audio, SAMPLE_RATE) if FeatureGroup.rhythm in groups else None
    )
    spectral = (
        compute_spectral(audio, SAMPLE_RATE)
        if FeatureGroup.spectral in groups
        else None
    )
    tonal = compute_tonal(audio, SAMPLE_RATE) if FeatureGroup.tonal in groups else None

    return TrackFeatures(
        song_id=song_id,
        rhythm=rhythm,
        spectral=spectral,
        tonal=tonal,
    )


def _check_feature_groups(
    groups: Sequence[FeatureGroup],
) -> tuple[FeatureGroup, ...]:
    feature_groups = tuple(groups)
    unknown = [group for group in feature_groups if group not in ALL_FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature groups: {unknown}; valid: {list(ALL_FEATURE_GROUPS)}"
        )
    return feature_groups


def _resolve_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return os.cpu_count() or 4
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {num_workers}")
    return num_workers


def _check_columns(table: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(table.columns)
    if missing:
        raise ValueError(f"track table missing columns: {sorted(missing)}")


def _init_worker() -> None:
    import essentia

    essentia.log.warningActive = False
