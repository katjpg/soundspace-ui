from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from soundspace.config.pipeline import LeidenConfig


@dataclass(frozen=True, slots=True)
class LeidenResult:
    membership: NDArray[np.int64]
    modularity: float
    n_communities: int


def leiden_partition(adjacency: sp.csr_matrix, config: LeidenConfig) -> LeidenResult:
    _validate(adjacency)
    graph = _to_igraph(adjacency, config.use_weights)
    partition = _run_leiden(graph, config)

    membership = np.asarray(partition.membership, dtype=np.int64)
    modularity = partition.modularity
    if modularity is None:
        raise ValueError("leidenalg returned modularity=None")

    return LeidenResult(
        membership=membership,
        modularity=float(modularity),
        n_communities=int(np.unique(membership).size),
    )


def _validate(adjacency: sp.csr_matrix) -> None:
    shape = adjacency.shape
    if shape[0] != shape[1]:
        raise ValueError(f"adjacency must be square, got shape {shape}")
    if adjacency.nnz == 0:
        raise ValueError("adjacency must have at least one edge")
    if not np.isfinite(adjacency.data).all():
        raise ValueError("adjacency contains non-finite weights")
    if float(adjacency.data.min()) < 0.0:
        raise ValueError("adjacency must have non-negative weights")


def _to_igraph(adjacency: sp.csr_matrix, use_weights: bool):
    import igraph as ig

    upper = sp.triu(adjacency, k=1).tocoo()
    edges = list(zip(upper.row.tolist(), upper.col.tolist()))
    graph = ig.Graph(n=int(adjacency.shape[0]), edges=edges, directed=False)
    if use_weights:
        graph.es["weight"] = upper.data.tolist()
    return graph


def _run_leiden(graph, config: LeidenConfig):
    import leidenalg

    weights = "weight" if config.use_weights else None
    if config.seed is None:
        return leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=float(config.resolution),
            n_iterations=int(config.n_iterations),
        )
    return leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=float(config.resolution),
        n_iterations=int(config.n_iterations),
        seed=int(config.seed),
    )
