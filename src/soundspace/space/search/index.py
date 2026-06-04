from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_L2_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class SearchIndex:
    track_ids: np.ndarray
    embeddings: np.ndarray
    mean: np.ndarray
    center: bool
    normalize: bool
    name: str

    @property
    def n_tracks(self) -> int:
        return len(self.track_ids)

    @property
    def embed_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def index_of(self, track_id: str) -> int | None:
        matches = np.flatnonzero(self.track_ids.astype(str) == str(track_id))
        return int(matches[0]) if matches.size else None

    @classmethod
    def load(cls, path: str | Path) -> SearchIndex:
        data = np.load(path, allow_pickle=False)
        embeddings = _l2_normalize_rows(
            np.asarray(data["embeddings"], dtype=np.float32)
        )
        return cls(
            track_ids=np.asarray(data["track_ids"]),
            embeddings=embeddings,
            mean=np.asarray(data["mean"], dtype=np.float32),
            center=bool(data["center"]) if "center" in data else True,
            normalize=bool(data["normalize"]) if "normalize" in data else True,
            name=str(data["name"]) if "name" in data else Path(path).stem,
        )


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if float(norms.min()) <= _L2_EPS:
        raise ValueError("zero-norm embedding in index")
    return (matrix / norms).astype(np.float32)
