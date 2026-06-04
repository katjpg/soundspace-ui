import copy
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from soundspace.config.dataset import DatasetConfig, load_dataset_config
from soundspace.config.pipeline import PipelineConfig, load_pipeline_config
from soundspace.config.settings import get_settings
from soundspace.dataset.tracks import read_tracks
from soundspace.space.evaluation.cluster import score_cluster_quality
from soundspace.space.evaluation.coherence import score_semantic_quality
from soundspace.space.evaluation.embed import check_embedding_sanity
from soundspace.space.evaluation.manifold import score_projection_quality
from soundspace.space.evaluation.topology import score_graph_quality
from soundspace.space.pipeline.cluster import load_graph

log = logging.getLogger(__name__)
console = Console()

_EMBEDDINGS_SUBDIR = "embeddings"

_OVERVIEW = """\
Evaluate the embedding space and its communities.

Each command scores one dimension of quality from artifacts written by the
embed and cluster stages, prints a summary, and merges the result into
data/artifacts/<model>_eval.json. Run 'eval all' for every dimension at once.
"""

app = typer.Typer(
    help=_OVERVIEW,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

_MODEL_OPTION = typer.Option(
    None,
    "-m",
    "--model",
    metavar="NAME",
    help="Model whose artifacts to evaluate. Defaults to the active model in config.",
)


def _config() -> DatasetConfig:
    settings = get_settings()
    raw = copy.deepcopy(settings.config())
    raw["data"]["root"] = str(settings.resolve_path(raw["data"]["root"]))
    return load_dataset_config(raw)


def _pipeline_config() -> PipelineConfig:
    settings = get_settings()
    raw = copy.deepcopy(settings.config())
    return load_pipeline_config(raw)


def _resolve_model(model: str | None) -> str:
    pipe = _pipeline_config()
    name = model or pipe.active
    if name not in pipe.embeddings:
        known = ", ".join(sorted(pipe.embeddings))
        raise typer.BadParameter(f"unknown model {name!r}; choices: {known}")
    return name


def _load_embeddings(
    dataset: DatasetConfig, model: str
) -> tuple[np.ndarray, list[str]]:
    path = dataset.processed_dir / _EMBEDDINGS_SUBDIR / f"{model}.npz"
    if not path.exists():
        raise typer.BadParameter(
            f"embeddings not found: {path}; run 'soundspace pipeline embed -m {model}' first"
        )
    data = np.load(path, allow_pickle=False)
    return data["embeddings"], [str(track_id) for track_id in data["track_ids"]]


def _load_clusters(dataset: DatasetConfig, model: str) -> pd.DataFrame:
    path = dataset.artifacts_dir / f"{model}_clusters.csv"
    if not path.exists():
        raise typer.BadParameter(
            f"clusters not found: {path}; run 'soundspace pipeline cluster -m {model}' first"
        )
    return pd.read_csv(path, dtype={"song_id": str})


def _load_graph(dataset: DatasetConfig, model: str):
    path = dataset.artifacts_dir / f"{model}_graph.npz"
    if not path.exists():
        raise typer.BadParameter(
            f"graph not found: {path}; run 'soundspace pipeline cluster -m {model}' first"
        )
    return load_graph(path)


def _eval_path(dataset: DatasetConfig, model: str) -> Path:
    return dataset.artifacts_dir / f"{model}_eval.json"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_clean(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    return value


def _write_section(
    dataset: DatasetConfig, model: str, name: str, section: dict
) -> Path:
    path = _eval_path(dataset, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    if path.exists():
        metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics[name] = _clean(section)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return path


def _metric_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right")
    for field, value in rows:
        table.add_row(field, value)
    console.print(table)


def _run_embed(dataset: DatasetConfig, model: str) -> None:
    embeddings, _ = _load_embeddings(dataset, model)
    sanity = check_embedding_sanity(embeddings)
    _metric_table(
        f"{model} embedding sanity",
        [
            ("samples", str(sanity.n_samples)),
            ("dims", str(sanity.n_dims)),
            ("has_nan", str(sanity.has_nan)),
            ("has_inf", str(sanity.has_inf)),
            ("has_zero_norm", str(sanity.has_zero_norm)),
            ("mean pairwise cosine", f"{sanity.mean_pairwise_cosine:.4f}"),
            ("std pairwise cosine", f"{sanity.std_pairwise_cosine:.4f}"),
        ],
    )
    _write_section(dataset, model, "embed", asdict(sanity))


def _run_cluster(dataset: DatasetConfig, model: str) -> None:
    embeddings, _ = _load_embeddings(dataset, model)
    _, membership = _load_graph(dataset, model)
    quality = score_cluster_quality(embeddings, membership, metric="cosine")
    _metric_table(
        f"{model} cluster quality",
        [
            ("clusters", str(quality.n_clusters)),
            ("silhouette (cosine)", f"{quality.silhouette_mean:.4f}"),
            ("size balance (CV)", f"{quality.size_balance:.4f}"),
            ("largest", str(int(max(quality.cluster_sizes, default=0)))),
            ("smallest", str(int(min(quality.cluster_sizes, default=0)))),
        ],
    )
    section = asdict(quality)
    section["size_balance"] = quality.size_balance
    _write_section(dataset, model, "cluster", section)


def _run_coherence(dataset: DatasetConfig, model: str) -> None:
    clusters = _load_clusters(dataset, model)
    tracks = read_tracks(dataset)
    merged = clusters[["song_id", "community"]].merge(tracks, on="song_id", how="left")
    membership = merged["community"].to_numpy()
    quality = score_semantic_quality(merged, membership)
    _metric_table(
        f"{model} semantic coherence",
        [
            ("communities", str(quality.n_labels)),
            ("mean style entropy", f"{quality.mean_style_entropy:.4f}"),
            ("mean theme entropy", f"{quality.mean_theme_entropy:.4f}"),
            ("mean mood entropy", f"{quality.mean_mood_entropy:.4f}"),
            ("mean V-A spread", f"{quality.mean_va_spread:.4f}"),
            ("mean quadrant coverage", f"{quality.mean_quadrant_coverage:.4f}"),
        ],
    )
    _write_section(dataset, model, "coherence", asdict(quality))


def _run_manifold(dataset: DatasetConfig, model: str) -> None:
    embeddings, track_ids = _load_embeddings(dataset, model)
    clusters = _load_clusters(dataset, model).set_index("song_id")
    coords = clusters.reindex(track_ids)[["x", "y"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all():
        raise typer.BadParameter(
            "cluster coords missing for some tracks; re-run cluster for this model"
        )
    quality = score_projection_quality(embeddings, coords, k=15, metric_high="cosine")
    _metric_table(
        f"{model} projection quality",
        [
            ("trustworthiness", f"{quality.trustworthiness:.4f}"),
            ("continuity", f"{quality.continuity:.4f}"),
            ("shepard rho", f"{quality.shepard_rho:.4f}"),
            ("k", str(quality.k)),
            ("samples scored", str(quality.n_samples)),
        ],
    )
    _write_section(dataset, model, "manifold", asdict(quality))


def _run_topology(dataset: DatasetConfig, model: str) -> None:
    adjacency, membership = _load_graph(dataset, model)
    cluster_cfg = _pipeline_config().cluster
    quality = score_graph_quality(
        adjacency,
        membership,
        resolution=cluster_cfg.leiden.resolution,
        symmetrize=cluster_cfg.knn.symmetrize,
        symmetrize_mode=cluster_cfg.knn.symmetrize_mode,
    )
    _metric_table(
        f"{model} graph quality",
        [
            ("nodes", str(quality.n_nodes)),
            ("edges", str(quality.n_edges)),
            ("components", str(quality.n_components)),
            ("modularity", f"{quality.modularity:.4f}"),
            ("communities", str(quality.n_communities)),
            ("mean degree", f"{quality.mean_degree:.2f}"),
        ],
    )
    _write_section(dataset, model, "topology", asdict(quality))


@app.command(
    help="Embedding matrix sanity: NaN/inf/zero-norm and pairwise cosine spread."
)
def embed(model: str | None = _MODEL_OPTION) -> None:
    dataset = _config()
    name = _resolve_model(model)
    _run_embed(dataset, name)
    typer.echo(f"Metrics: {_eval_path(dataset, name)}")


@app.command(help="Geometric cluster quality: cosine silhouette and size balance.")
def cluster(model: str | None = _MODEL_OPTION) -> None:
    dataset = _config()
    name = _resolve_model(model)
    _run_cluster(dataset, name)
    typer.echo(f"Metrics: {_eval_path(dataset, name)}")


@app.command(help="Semantic coherence: per-community tag entropy and V-A spread.")
def coherence(model: str | None = _MODEL_OPTION) -> None:
    dataset = _config()
    name = _resolve_model(model)
    _run_coherence(dataset, name)
    typer.echo(f"Metrics: {_eval_path(dataset, name)}")


@app.command(help="UMAP projection fidelity: trustworthiness, continuity, Shepard.")
def manifold(model: str | None = _MODEL_OPTION) -> None:
    dataset = _config()
    name = _resolve_model(model)
    _run_manifold(dataset, name)
    typer.echo(f"Metrics: {_eval_path(dataset, name)}")


@app.command(help="kNN graph structure: modularity, components, mean degree.")
def topology(model: str | None = _MODEL_OPTION) -> None:
    dataset = _config()
    name = _resolve_model(model)
    _run_topology(dataset, name)
    typer.echo(f"Metrics: {_eval_path(dataset, name)}")


@app.command(
    name="all", help="Run every evaluation dimension and write the full metrics file."
)
def run_all(model: str | None = _MODEL_OPTION) -> None:
    dataset = _config()
    name = _resolve_model(model)
    _run_embed(dataset, name)
    _run_topology(dataset, name)
    _run_cluster(dataset, name)
    _run_coherence(dataset, name)
    _run_manifold(dataset, name)
    typer.echo(f"Metrics: {_eval_path(dataset, name)}")
