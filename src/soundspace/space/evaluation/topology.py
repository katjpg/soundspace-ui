from dataclasses import dataclass
from typing import cast

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags

from soundspace.config.pipeline import SymmetrizeMode

_WEIGHT_ATTR = "weight"


@dataclass(frozen=True, slots=True)
class GraphQuality:
    n_nodes: int
    n_edges: int
    n_components: int
    modularity: float
    n_communities: int
    community_sizes: np.ndarray
    mean_degree: float


def score_graph_quality(
    adjacency: csr_matrix,
    membership: np.ndarray,
    *,
    resolution: float = 1.0,
    symmetrize: bool = True,
    symmetrize_mode: SymmetrizeMode = "max",
) -> GraphQuality:
    if membership.ndim != 1:
        raise ValueError(f"membership must be 1D, got shape {membership.shape}")

    adj = _to_csr(adjacency)
    n = int(adj.shape[0])
    if len(membership) != n:
        raise ValueError(
            f"membership length ({len(membership)}) must match adjacency size ({n})"
        )
    if n < 2:
        return GraphQuality(n, 0, 0, 0.0, 0, np.array([], dtype=np.int64), 0.0)

    adj = _prepare_adjacency(adj, symmetrize=symmetrize, mode=symmetrize_mode)
    graph = nx.from_scipy_sparse_array(
        adj, create_using=nx.Graph, edge_attribute=_WEIGHT_ATTR
    )

    degrees = np.array([d for _, d in graph.degree()], dtype=np.float64)
    mean_degree = float(np.mean(degrees)) if len(degrees) > 0 else 0.0

    unique_labels = np.unique(membership)
    n_communities = len(unique_labels)
    community_sizes = np.array(
        [int(np.sum(membership == lab)) for lab in unique_labels], dtype=np.int64
    )

    if n_communities < 2:
        modularity = 0.0
    else:
        communities = [
            set(np.flatnonzero(membership == lab).tolist()) for lab in unique_labels
        ]
        has_weights = adj.nnz > 0 and bool(np.any(adj.data != 1.0))
        modularity = float(
            nx.algorithms.community.modularity(
                graph,
                communities,
                weight=_WEIGHT_ATTR if has_weights else None,
                resolution=resolution,
            )
        )

    return GraphQuality(
        n_nodes=n,
        n_edges=graph.number_of_edges(),
        n_components=nx.number_connected_components(graph),
        modularity=modularity,
        n_communities=n_communities,
        community_sizes=community_sizes,
        mean_degree=mean_degree,
    )


def _to_csr(adjacency: csr_matrix) -> csr_matrix:
    adj = sp.csr_matrix(adjacency)
    shape = adj.shape
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"adjacency must be square 2D, got shape {shape}")
    return adj


def _prepare_adjacency(
    adj: csr_matrix, *, symmetrize: bool, mode: SymmetrizeMode
) -> csr_matrix:
    diag = adj.diagonal()
    if np.any(diag != 0):
        adj = cast(csr_matrix, (adj - diags(diag, offsets=0, format="csr")).tocsr())
        adj.eliminate_zeros()

    if symmetrize:
        diff = adj - adj.T
        diff.eliminate_zeros()
        if diff.nnz > 0:
            if mode == "max":
                adj = cast(csr_matrix, adj.maximum(adj.T).tocsr())
            elif mode == "min":
                adj = cast(csr_matrix, adj.minimum(adj.T).tocsr())
            else:
                adj = cast(csr_matrix, ((adj + adj.T) * 0.5).tocsr())
            adj.eliminate_zeros()

    return adj
