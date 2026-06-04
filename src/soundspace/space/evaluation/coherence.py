from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from soundspace.dataset.tags import split_tags

_STYLE_COL = "style"
_THEME_COL = "theme"
_MOOD_COL = "mood_all"
_AROUSAL_COL = "arousal"
_VALENCE_COL = "valence"


@dataclass(frozen=True, slots=True)
class LabelCoherence:
    label: int
    size: int
    style_entropy: float
    theme_entropy: float
    mood_entropy: float
    va_spread: float
    quadrant_coverage: float


@dataclass(frozen=True, slots=True)
class SemanticQuality:
    n_labels: int
    n_samples: int
    mean_style_entropy: float
    mean_theme_entropy: float
    mean_mood_entropy: float
    mean_va_spread: float
    mean_quadrant_coverage: float
    labels: list[LabelCoherence]


def score_semantic_quality(
    table: pd.DataFrame,
    membership: np.ndarray,
) -> SemanticQuality:
    if len(table) != len(membership):
        raise ValueError(
            f"table rows ({len(table)}) must match membership length ({len(membership)})"
        )

    n_samples = len(table)
    if n_samples == 0:
        return SemanticQuality(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    frame = table.reset_index(drop=True)
    results: list[LabelCoherence] = []

    for lab in np.unique(membership):
        mask = membership == lab
        sub = frame[mask]
        if len(sub) == 0:
            continue

        arousals = sub[_AROUSAL_COL].astype(float).tolist()
        valences = sub[_VALENCE_COL].astype(float).tolist()

        results.append(
            LabelCoherence(
                label=int(lab),
                size=len(sub),
                style_entropy=_tag_entropy(_tag_lists(sub, _STYLE_COL)),
                theme_entropy=_tag_entropy(_tag_lists(sub, _THEME_COL)),
                mood_entropy=_tag_entropy(_tag_lists(sub, _MOOD_COL)),
                va_spread=_va_spread(arousals, valences),
                quadrant_coverage=_quadrant_coverage(arousals, valences),
            )
        )

    if not results:
        return SemanticQuality(0, n_samples, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    return SemanticQuality(
        n_labels=len(results),
        n_samples=n_samples,
        mean_style_entropy=float(np.mean([r.style_entropy for r in results])),
        mean_theme_entropy=float(np.mean([r.theme_entropy for r in results])),
        mean_mood_entropy=float(np.mean([r.mood_entropy for r in results])),
        mean_va_spread=float(np.mean([r.va_spread for r in results])),
        mean_quadrant_coverage=float(np.mean([r.quadrant_coverage for r in results])),
        labels=results,
    )


def _tag_lists(sub: pd.DataFrame, column: str) -> list[tuple[str, ...]]:
    if column not in sub.columns:
        return []
    return [tuple(split_tags(cell)) for cell in sub[column]]


def _tag_entropy(tag_lists: Sequence[tuple[str, ...]]) -> float:
    counter: Counter[str] = Counter()
    for tags in tag_lists:
        counter.update(tags)
    counts = list(counter.values())
    if not counts:
        return 0.0

    total = sum(counts)
    nonzero = [c for c in counts if c > 0]
    if total == 0 or len(nonzero) <= 1:
        return 0.0

    probs = np.array([c / total for c in nonzero], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(len(nonzero)))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _va_spread(arousals: Sequence[float], valences: Sequence[float]) -> float:
    if len(arousals) < 2:
        return 0.0
    a = np.array(arousals, dtype=np.float64)
    v = np.array(valences, dtype=np.float64)
    distances = np.sqrt((a - a.mean()) ** 2 + (v - v.mean()) ** 2)
    return float(np.std(distances))


def _quadrant_coverage(arousals: Sequence[float], valences: Sequence[float]) -> float:
    if len(arousals) == 0:
        return 0.0
    present: set[int] = set()
    for a, v in zip(arousals, valences):
        if a >= 0 and v >= 0:
            present.add(1)
        elif a >= 0 and v < 0:
            present.add(2)
        elif a < 0 and v < 0:
            present.add(3)
        else:
            present.add(4)
    return len(present) / 4.0
