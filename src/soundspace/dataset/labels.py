import json
from pathlib import Path

import pandas as pd

from soundspace.dataset.tags import split_tags

LABEL_SOURCE_COLUMNS: dict[str, str] = {
    "Moods": "mood",
    "MoodsAll": "mood_all",
    "Genres": "genre",
    "Themes": "theme",
    "Styles": "style",
}


def build_labels(metadata_path: str | Path, out_path: str | Path) -> Path:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")

    metadata = pd.read_csv(path, dtype=str)
    missing = set(LABEL_SOURCE_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    labels = {
        name: _collect_labels(metadata[raw])
        for raw, name in LABEL_SOURCE_COLUMNS.items()
    }

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def _collect_labels(column: pd.Series) -> list[str]:
    labels: set[str] = set()
    for cell in column:
        labels.update(split_tags(cell))
    return sorted(labels, key=str.casefold)
