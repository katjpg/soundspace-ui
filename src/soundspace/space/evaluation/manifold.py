from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

_DEFAULT_SUBSAMPLE_N = 5000


@dataclass(frozen=True, slots=True)
class ProjectionQuality:
    trustworthiness: float
    continuity: float
    shepard_rho: float
    k: int
    n_samples: int


def score_projection_quality(
    x_high: np.ndarray,
    x_low: np.ndarray,
    *,
    k: int = 15,
    metric_high: str = "cosine",
    metric_low: str = "euclidean",
    subsample_n: int | None = _DEFAULT_SUBSAMPLE_N,
    seed: int = 42,
) -> ProjectionQuality:
    if x_high.ndim != 2 or x_low.ndim != 2:
        raise ValueError("x_high and x_low must be 2D arrays")
    if len(x_high) != len(x_low):
        raise ValueError(
            f"shape mismatch: x_high has {len(x_high)} rows, x_low has {len(x_low)} rows"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(x_high) < 2:
        return ProjectionQuality(0.0, 0.0, 0.0, k, len(x_high))

    high, low = _subsample(x_high, x_low, subsample_n=subsample_n, seed=seed)
    n = len(high)
    if n <= k:
        return ProjectionQuality(0.0, 0.0, 0.0, k, n)

    k_eval = max(1, min(k, (n // 2) - 1))
    dist_high = pairwise_distances(high, metric=metric_high).astype(
        np.float64, copy=False
    )
    dist_low = pairwise_distances(low, metric=metric_low).astype(np.float64, copy=False)

    nn_high = _knn_from_distances(dist_high, k=k_eval)
    nn_low = _knn_from_distances(dist_low, k=k_eval)
    inv_rank_high = _inverse_ranks(dist_high)
    inv_rank_low = _inverse_ranks(dist_low)

    trust, cont = _trustworthiness_continuity(
        nn_high, nn_low, inv_rank_high, inv_rank_low, n=n, k=k_eval
    )
    rho = _shepard_correlation(dist_high, dist_low, seed=seed)

    return ProjectionQuality(trust, cont, rho, k_eval, n)


def _subsample(
    x_high: np.ndarray, x_low: np.ndarray, *, subsample_n: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x_high)
    if subsample_n is None or subsample_n >= n:
        return np.asarray(x_high, np.float64), np.asarray(x_low, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=subsample_n, replace=False)
    return np.asarray(x_high[idx], np.float64), np.asarray(x_low[idx], np.float64)


def _knn_from_distances(dist: np.ndarray, *, k: int) -> np.ndarray:
    n = len(dist)
    masked = dist.copy()
    np.fill_diagonal(masked, np.inf)
    idx = np.argpartition(masked, kth=k - 1, axis=1)[:, :k]
    row = np.arange(n)[:, None]
    order = np.argsort(masked[row, idx], axis=1)
    return idx[row, order].astype(np.int64, copy=False)


def _inverse_ranks(dist: np.ndarray) -> np.ndarray:
    n = len(dist)
    masked = dist.copy()
    np.fill_diagonal(masked, -1.0)
    order = np.argsort(masked, axis=1)
    inv = np.empty_like(order, dtype=np.int64)
    row = np.arange(n)[:, None]
    inv[row, order] = np.arange(n)[None, :]
    return inv


def _trustworthiness_continuity(
    nn_high: np.ndarray,
    nn_low: np.ndarray,
    inv_rank_high: np.ndarray,
    inv_rank_low: np.ndarray,
    *,
    n: int,
    k: int,
) -> tuple[float, float]:
    denom = float(n * k * (2 * n - 3 * k - 1))
    if denom <= 0:
        return 0.0, 0.0

    trust_sum = 0.0
    cont_sum = 0.0
    for i in range(n):
        high_set = set(int(x) for x in nn_high[i].tolist())
        low_set = set(int(x) for x in nn_low[i].tolist())
        for j in nn_low[i]:
            if int(j) not in high_set:
                trust_sum += float(int(inv_rank_high[i, int(j)]) - k)
        for j in nn_high[i]:
            if int(j) not in low_set:
                cont_sum += float(int(inv_rank_low[i, int(j)]) - k)

    factor = 2.0 / denom
    return float(1.0 - factor * trust_sum), float(1.0 - factor * cont_sum)


def _shepard_correlation(
    dist_high: np.ndarray, dist_low: np.ndarray, *, seed: int, n_pairs: int = 5000
) -> float:
    n = len(dist_high)
    if n < 2:
        return 0.0
    n_pairs_eff = min(n_pairs, (n * (n - 1)) // 2)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs_eff)
    j = rng.integers(0, n, size=n_pairs_eff)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(np.sum(same)))
        same = i == j

    result = spearmanr(dist_high[i, j], dist_low[i, j])
    rho = (
        float(result[0]) if hasattr(result, "__getitem__") else float(result.statistic)
    )
    return 0.0 if np.isnan(rho) else rho
