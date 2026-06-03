from collections.abc import Sequence

import pandas as pd

from soundspace.types import TagGroup

_TRACK_COLUMNS: tuple[str, ...] = (
    "song_id",
    "quadrant",
    "arousal",
    "valence",
    "artist",
    "title",
    "duration",
)

_TAG_COLUMNS: dict[TagGroup, tuple[str, ...]] = {
    "mood": ("mood", "mood_all"),
    "genre": ("genre", "genre_main"),
    "theme": ("theme",),
    "style": ("style",),
}

_TAG_WEIGHT_COLUMNS: dict[TagGroup, tuple[str, ...]] = {
    "mood": ("mood_all_weights",),
    "genre": ("genre_weights",),
    "theme": ("theme_weights",),
    "style": ("style_weights",),
}


def select_columns(
    table: pd.DataFrame,
    groups: Sequence[TagGroup],
    *,
    include_weights: bool = True,
) -> pd.DataFrame:
    unknown = [group for group in groups if group not in _TAG_COLUMNS]
    if unknown:
        raise ValueError(
            f"unknown tag groups: {unknown}; valid groups: {sorted(_TAG_COLUMNS)}"
        )

    columns: list[str] = list(_TRACK_COLUMNS)
    seen: set[TagGroup] = set()
    for group in groups:
        if group in seen:
            continue
        seen.add(group)
        columns.extend(_TAG_COLUMNS[group])
        if include_weights:
            columns.extend(_TAG_WEIGHT_COLUMNS[group])

    _check_columns(table, columns)
    return table[columns].copy()


def _check_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"table is missing columns: {missing}")
