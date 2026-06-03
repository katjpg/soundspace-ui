from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from soundspace.types import TagGroup

TAG_GROUP_COLUMNS: dict[TagGroup, str] = {
    "mood": "mood_all",
    "genre": "genre_main",
    "theme": "theme",
    "style": "style",
}


@dataclass(frozen=True, slots=True)
class FilterConfig:
    required_groups: tuple[TagGroup, ...] = ("mood", "genre", "theme", "style")
    sample_size: int | None = None
    seed: int = 7

    def __post_init__(self) -> None:
        unknown = set(self.required_groups) - set(TAG_GROUP_COLUMNS)
        if unknown:
            raise ValueError(
                f"unknown tag groups: {sorted(unknown)}; "
                f"valid groups: {sorted(TAG_GROUP_COLUMNS)}"
            )
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError(f"sample_size must be > 0, got {self.sample_size}")


@dataclass(frozen=True, slots=True)
class FilterStats:
    total: int
    dropped: dict[str, int]
    kept_before_sample: int
    kept: int

    def summary(self) -> str:
        line = f"kept {self.kept_before_sample}/{self.total}"
        dropped = self.total - self.kept_before_sample
        if dropped:
            reasons = ", ".join(
                f"{count} {reason}" for reason, count in self.dropped.items()
            )
            line += f"; dropped {dropped} ({reasons})"
        if self.kept != self.kept_before_sample:
            line += f"; sampled to {self.kept}"
        return line


def filter_tracks(
    table: pd.DataFrame,
    audio_dir: str | Path,
    config: FilterConfig | None = None,
) -> tuple[pd.DataFrame, FilterStats]:
    cfg = config or FilterConfig()
    root = Path(audio_dir)
    if not root.exists():
        raise FileNotFoundError(f"audio dir not found: {root}")

    _check_columns(table, cfg.required_groups)

    total = len(table)
    dropped: dict[str, int] = {}
    tracks = table

    for group in cfg.required_groups:
        column = TAG_GROUP_COLUMNS[group]
        missing_tag = tracks[column].fillna("").str.strip() == ""
        count = int(missing_tag.sum())
        if count:
            dropped[f"missing {group}"] = count
        tracks = tracks[~missing_tag]

    has_audio = pd.Series(
        [
            (root / quadrant / f"{song_id}.mp3").exists()
            for song_id, quadrant in zip(tracks["song_id"], tracks["quadrant"])
        ],
        index=tracks.index,
        dtype=bool,
    )
    count = int((~has_audio).sum())
    if count:
        dropped["missing audio"] = count
    tracks = tracks[has_audio]

    kept_before_sample = len(tracks)
    if cfg.sample_size is not None:
        tracks = _sample_tracks(tracks, cfg.sample_size, cfg.seed)

    stats = FilterStats(
        total=total,
        dropped=dropped,
        kept_before_sample=kept_before_sample,
        kept=len(tracks),
    )
    return tracks.reset_index(drop=True), stats


def _sample_tracks(table: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size >= len(table):
        return table
    return table.sample(n=size, random_state=seed).sort_index()


def _check_columns(table: pd.DataFrame, groups: tuple[TagGroup, ...]) -> None:
    required = {"song_id", "quadrant"}
    required.update(TAG_GROUP_COLUMNS[group] for group in groups)
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"track table is missing columns: {sorted(missing)}")
