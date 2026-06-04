from dataclasses import dataclass

import numpy as np

_L2_EPS = 1e-12
_PAIRWISE_SAMPLE_SIZE = 10_000
_RANDOM_SEED = 42


@dataclass(frozen=True, slots=True)
class EmbeddingSanity:
    n_samples: int
    n_dims: int
    has_nan: bool
    has_inf: bool
    has_zero_norm: bool
    mean_pairwise_cosine: float
    std_pairwise_cosine: float


def check_embedding_sanity(
    embeddings: np.ndarray,
    *,
    n_pairs: int = _PAIRWISE_SAMPLE_SIZE,
    seed: int = _RANDOM_SEED,
    eps: float = _L2_EPS,
) -> EmbeddingSanity:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")

    matrix = np.asarray(embeddings, dtype=np.float64)
    n_samples, n_dims = matrix.shape

    if n_samples == 0:
        return EmbeddingSanity(0, n_dims, False, False, False, 0.0, 0.0)

    norms = np.linalg.norm(matrix, axis=1)
    mean_cos, std_cos = _sample_pairwise_cosine(
        matrix, n_pairs=n_pairs, seed=seed, eps=eps
    )

    return EmbeddingSanity(
        n_samples=n_samples,
        n_dims=n_dims,
        has_nan=bool(np.any(np.isnan(matrix))),
        has_inf=bool(np.any(np.isinf(matrix))),
        has_zero_norm=bool(np.any(norms <= eps)),
        mean_pairwise_cosine=mean_cos,
        std_pairwise_cosine=std_cos,
    )


def _sample_pairwise_cosine(
    matrix: np.ndarray, *, n_pairs: int, seed: int, eps: float
) -> tuple[float, float]:
    n = len(matrix)
    if n < 2:
        return 0.0, 0.0

    unit = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + eps)
    n_actual = min(n_pairs, (n * (n - 1)) // 2)

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_actual)
    j = rng.integers(0, n, size=n_actual)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(np.sum(same)))
        same = i == j

    sims = np.einsum("ij,ij->i", unit[i], unit[j])
    return float(np.mean(sims)), float(np.std(sims))
