import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from soundspace.config.pipeline import KNNConfig, SymmetrizeMode

_DTYPE = np.float32


def build_knn(embeddings: np.ndarray, config: KNNConfig) -> sp.csr_matrix:
    _validate(embeddings, config.k)
    indices, distances = _knn_cosine(embeddings, config.k)
    directed = _directed_affinity(indices, distances, len(embeddings))
    if not config.symmetrize:
        return directed
    return _symmetrize(directed, config.symmetrize_mode)


def _validate(embeddings: np.ndarray, k: int) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    n_nodes = int(embeddings.shape[0])
    if n_nodes < 2:
        raise ValueError("embeddings must contain at least 2 rows")
    if not 0 < k < n_nodes:
        raise ValueError(f"k must be in (0, {n_nodes}), got {k}")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")


def _knn_cosine(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="auto")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)
    return indices[:, 1:], distances[:, 1:]


def _directed_affinity(
    indices: np.ndarray, distances: np.ndarray, n_nodes: int
) -> sp.csr_matrix:
    k = int(indices.shape[1])
    rows = np.repeat(np.arange(n_nodes, dtype=np.int32), k)
    cols = indices.reshape(-1).astype(np.int32, copy=False)
    affinity = np.clip(1.0 - distances.reshape(-1), 0.0, 1.0).astype(_DTYPE)

    directed = sp.csr_matrix(
        (affinity, (rows, cols)), shape=(n_nodes, n_nodes), dtype=_DTYPE
    )
    directed.sum_duplicates()
    directed.setdiag(0.0)
    directed.eliminate_zeros()
    return directed


def _symmetrize(directed: sp.csr_matrix, mode: SymmetrizeMode) -> sp.csr_matrix:
    if mode == "max":
        undirected = directed.maximum(directed.T)
    elif mode == "min":
        undirected = directed.minimum(directed.T)
    else:
        undirected = (directed + directed.T) * 0.5

    undirected = undirected.tocsr()
    undirected.sum_duplicates()
    undirected.setdiag(0.0)
    undirected.eliminate_zeros()
    return undirected
