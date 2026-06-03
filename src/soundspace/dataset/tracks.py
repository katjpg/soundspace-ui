from pathlib import Path

import pandas as pd

from soundspace.config.dataset import DatasetConfig


def read_tracks(dataset: DatasetConfig) -> pd.DataFrame:
    table = pd.read_csv(
        dataset.processed_dir / f"{dataset.active}.csv", dtype={"song_id": str}
    )
    if "song_id" not in table.columns:
        raise ValueError("dataset table missing 'song_id' column")
    return table


def resolve_audio_paths(table: pd.DataFrame, audio_dir: Path) -> dict[str, Path]:
    if "song_id" not in table.columns:
        raise ValueError("track table missing 'song_id' column")

    has_audio_path = "audio_path" in table.columns
    has_quadrant = "quadrant" in table.columns

    paths: dict[str, Path] = {}
    for row in table.to_dict("records"):
        song_id = str(row["song_id"])
        if has_audio_path and isinstance(row["audio_path"], str):
            candidate = Path(row["audio_path"])
            paths[song_id] = (
                candidate if candidate.is_absolute() else audio_dir / candidate
            )
        elif has_quadrant:
            paths[song_id] = audio_dir / str(row["quadrant"]) / f"{song_id}.mp3"
        else:
            paths[song_id] = audio_dir / f"{song_id}.mp3"
    return paths
