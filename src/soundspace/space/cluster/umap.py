import numpy as np

from soundspace.config.pipeline import UMAPConfig


def umap_layout(embeddings: np.ndarray, config: UMAPConfig) -> np.ndarray:
    _validate(embeddings, config)

    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(config.n_neighbors),
        min_dist=float(config.min_dist),
        metric=str(config.metric),
        random_state=None if config.seed is None else int(config.seed),
    )
    coords = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)

    if coords.ndim != 2 or int(coords.shape[1]) != 2:
        raise ValueError(f"UMAP returned unexpected shape {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("UMAP produced non-finite coordinates")
    return coords


def _validate(embeddings: np.ndarray, config: UMAPConfig) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    n_nodes = int(embeddings.shape[0])
    if n_nodes < 2:
        raise ValueError("embeddings must contain at least 2 rows")
    if not 1 < config.n_neighbors < n_nodes:
        raise ValueError(
            f"n_neighbors must be in (1, {n_nodes}), got {config.n_neighbors}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
