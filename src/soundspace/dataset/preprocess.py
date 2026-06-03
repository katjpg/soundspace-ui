import logging
from pathlib import Path

import pandas as pd

from soundspace.dataset.filter import FilterConfig, filter_tracks
from soundspace.dataset.tags import split_tags

log = logging.getLogger(__name__)

METADATA_COLUMNS: dict[str, str] = {
    "Song": "song_id",
    "Quadrant": "quadrant",
    "Artist": "artist",
    "Title": "title",
    "Duration": "duration",
    "Moods": "mood",
    "MoodsAll": "mood_all",
    "MoodsAllWeights": "mood_all_weights",
    "Genres": "genre",
    "GenreWeights": "genre_weights",
    "Themes": "theme",
    "ThemeWeights": "theme_weights",
    "Styles": "style",
    "StyleWeights": "style_weights",
}

AV_COLUMNS: dict[str, str] = {
    "Song": "song_id",
    "Arousal": "arousal",
    "Valence": "valence",
}

GENRE_TAXONOMY: dict[str, list[str]] = {
    "Rock/Pop": ["Pop/Rock"],
    "Electronic": ["Electronic", "Easy Listening"],
    "Hip-Hop/Rap": ["Rap"],
    "R&B/Soul": ["R&B"],
    "Jazz": ["Jazz"],
    "Classical": ["Classical"],
    "Folk/Country": ["Folk", "Country"],
    "Blues": ["Blues"],
    "World": ["International", "Latin", "Reggae"],
    "Experimental": ["Avant-Garde", "New Age"],
}

EXCLUDED_GENRES: tuple[str, ...] = (
    "Children's",
    "Holiday",
    "Religious",
    "Comedy/Spoken",
    "Stage & Screen",
    "Vocal",
)

_STANDARDIZED_TAGS: tuple[str, ...] = ("mood", "mood_all", "theme", "style")

TABLE_COLUMNS: list[str] = [
    "song_id",
    "quadrant",
    "arousal",
    "valence",
    "artist",
    "title",
    "duration",
    "mood",
    "mood_all",
    "mood_all_weights",
    "genre",
    "genre_weights",
    "genre_main",
    "theme",
    "theme_weights",
    "style",
    "style_weights",
]


def preprocess(
    metadata_path: str | Path,
    av_path: str | Path,
    audio_dir: str | Path,
    output_path: str | Path,
    *,
    config: FilterConfig | None = None,
) -> Path:
    """Write a cleaned MERGE dataset CSV.

    Joins metadata and arousal-valence tables by song_id, filters tracks,
    applies the output schema, and writes the result.

    Args:
        metadata_path: Raw MERGE metadata CSV.
        av_path: MERGE arousal-valence CSV.
        audio_dir: Directory containing audio files.
        output_path: Output CSV path.
        config: Track filter config.

    Returns:
        Path to the written CSV.
    """
    metadata = _read_metadata(Path(metadata_path))
    av_values = _read_av_values(Path(av_path))
    tracks = _merge_av(metadata, av_values)

    tracks = _clean_tracks(tracks)
    tracks, stats = filter_tracks(tracks, audio_dir, config)
    log.info(stats.summary())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracks[TABLE_COLUMNS].to_csv(output, index=False)
    log.info("wrote %d rows -> %s", len(tracks), output)
    return output


def _read_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")
    table = pd.read_csv(path, dtype=str)
    _require_columns(table, set(METADATA_COLUMNS), path)
    table = table.rename(columns=METADATA_COLUMNS)
    return table[list(METADATA_COLUMNS.values())]


def _read_av_values(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AV values file not found: {path}")
    table = pd.read_csv(path, dtype={"Song": str})
    _require_columns(table, set(AV_COLUMNS), path)
    table = table.rename(columns=AV_COLUMNS)
    return table[list(AV_COLUMNS.values())]


def _merge_av(metadata: pd.DataFrame, av_values: pd.DataFrame) -> pd.DataFrame:
    tracks = metadata.merge(av_values, on="song_id", how="inner", validate="one_to_one")
    if len(tracks) != len(metadata):
        raise ValueError(
            f"AV merge dropped {len(metadata) - len(tracks)} metadata rows"
        )
    return tracks


def _require_columns(table: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def _clean_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    tracks = tracks.copy()
    for column in _STANDARDIZED_TAGS:
        tracks[column] = tracks[column].map(_lower_tags)
    tracks["genre_main"] = tracks["genre"].map(_consolidate_genres)
    return tracks


def _lower_tags(value: object) -> str:
    return ", ".join(part.lower() for part in split_tags(value))


def _consolidate_genres(value: object) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for genre in split_tags(value):
        mapped = _map_genre(genre)
        if mapped is not None and mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return ", ".join(out)


def _map_genre(genre: str) -> str | None:
    if genre in EXCLUDED_GENRES:
        return None
    for consolidated, originals in GENRE_TAXONOMY.items():
        if genre in originals:
            return consolidated
    return None
