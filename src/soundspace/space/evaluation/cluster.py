from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score


@dataclass(frozen=True, slots=True)
class ClusterQuality:
    n_clusters: int
    cluster_sizes: np.ndarray
    silhouette_mean: float

    @property
    def size_balance(self) -> float:
        if len(self.cluster_sizes) == 0:
            return float("nan")
        mean_size = float(np.mean(self.cluster_sizes))
        if mean_size == 0:
            return float("nan")
        return float(np.std(self.cluster_sizes) / mean_size)


def score_cluster_quality(
    embeddings: np.ndarray,
    membership: np.ndarray,
    metric: str = "cosine",
) -> ClusterQuality:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if membership.ndim != 1:
        raise ValueError(f"membership must be 1D, got shape {membership.shape}")
    if len(embeddings) != len(membership):
        raise ValueError(
            f"embeddings rows ({len(embeddings)}) must match membership length ({len(membership)})"
        )

    n_samples = len(embeddings)
    if n_samples == 0:
        return ClusterQuality(0, np.array([], dtype=np.int64), float("nan"))

    unique_labels = np.unique(membership)
    n_clusters = len(unique_labels)
    cluster_sizes = np.array(
        [int(np.sum(membership == lab)) for lab in unique_labels], dtype=np.int64
    )

    if n_clusters < 2 or n_clusters >= n_samples:
        return ClusterQuality(n_clusters, cluster_sizes, float("nan"))

    try:
        sil = float(silhouette_score(embeddings, membership, metric=metric))
    except Exception:
        sil = float("nan")

    return ClusterQuality(n_clusters, cluster_sizes, sil)
