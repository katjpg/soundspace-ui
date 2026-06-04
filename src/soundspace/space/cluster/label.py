from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from soundspace.config.pipeline import LabelConfig
from soundspace.dataset.tags import split_tags

_COMMUNITY_COLUMN = "__community"


def label_communities(
    table: pd.DataFrame,
    song_ids: Sequence[str],
    membership: np.ndarray,
    config: LabelConfig,
) -> dict[int, str]:
    community_by_song = {
        str(song_id): int(community) for song_id, community in zip(song_ids, membership)
    }

    table = table.copy()
    table[_COMMUNITY_COLUMN] = table["song_id"].astype(str).map(community_by_song)
    table = table[table[_COMMUNITY_COLUMN].notna()]

    labels: dict[int, str] = {}
    for community, group in table.groupby(_COMMUNITY_COLUMN):
        labels[int(community)] = _top_tags(group, config.columns, config.top_n)
    return labels


def _top_tags(group: pd.DataFrame, columns: Sequence[str], top_n: int) -> str:
    counter: Counter[str] = Counter()
    for column in columns:
        if column not in group.columns:
            continue
        for cell in group[column]:
            counter.update(split_tags(cell))
    if not counter:
        return ""
    return ", ".join(tag for tag, _ in counter.most_common(top_n))
