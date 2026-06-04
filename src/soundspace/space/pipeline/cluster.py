from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from soundspace.config.dataset import DatasetConfig
from soundspace.config.pipeline import ClusterConfig
from soundspace.dataset.tracks import read_tracks
from soundspace.space.cluster.knn import build_knn
from soundspace.space.cluster.label import label_communities
from soundspace.space.cluster.leiden import leiden_partition
from soundspace.space.cluster.umap import umap_layout

_EMBEDDINGS_SUBDIR = "embeddings"


@dataclass(frozen=True, slots=True)
class ClusterResult:
    model: str
    csv_path: Path
    graph_path: Path
    n_tracks: int
    n_communities: int
    modularity: float
    n_edges: int


def cluster_dataset(
    dataset: DatasetConfig,
    config: ClusterConfig,
    *,
    model: str,
) -> ClusterResult:
    embeddings, song_ids = _load_embeddings(dataset, model)

    adjacency = build_knn(embeddings, config.knn)
    leiden = leiden_partition(adjacency, config.leiden)
    coords = umap_layout(embeddings, config.umap)

    table = read_tracks(dataset)
    labels = label_communities(table, song_ids, leiden.membership, config.label)
    frame = _build_frame(song_ids, leiden.membership, labels, coords)

    csv_path = dataset.artifacts_dir / f"{model}_clusters.csv"
    graph_path = dataset.artifacts_dir / f"{model}_graph.npz"
    _write_csv(frame, csv_path)
    _write_graph(graph_path, adjacency, leiden.membership)

    return ClusterResult(
        model=model,
        csv_path=csv_path,
        graph_path=graph_path,
        n_tracks=len(song_ids),
        n_communities=leiden.n_communities,
        modularity=leiden.modularity,
        n_edges=int(adjacency.nnz // 2),
    )


def load_graph(path: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    adjacency = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"]),
    )
    return adjacency, data["membership"]


def _load_embeddings(
    dataset: DatasetConfig, model: str
) -> tuple[np.ndarray, list[str]]:
    path = dataset.processed_dir / _EMBEDDINGS_SUBDIR / f"{model}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"embeddings not found: {path}; run 'soundspace pipeline embed -m {model}' first"
        )
    data = np.load(path, allow_pickle=False)
    return data["embeddings"], [str(track_id) for track_id in data["track_ids"]]


def _build_frame(
    song_ids: list[str],
    membership: np.ndarray,
    labels: dict[int, str],
    coords: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "song_id": song_ids,
            "community": membership.astype(np.int64),
            "label": [labels.get(int(c), "") for c in membership],
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_graph(path: Path, adjacency: sp.csr_matrix, membership: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    adj = adjacency.tocsr()
    np.savez_compressed(
        path,
        data=adj.data,
        indices=adj.indices,
        indptr=adj.indptr,
        shape=np.asarray(adj.shape),
        membership=membership.astype(np.int64),
    )
