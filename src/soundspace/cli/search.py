import copy
import logging
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from soundspace.config.dataset import DatasetConfig, load_dataset_config
from soundspace.config.pipeline import (
    EmbeddingConfig,
    PipelineConfig,
    load_pipeline_config,
)
from soundspace.config.settings import get_settings
from soundspace.dataset.tracks import read_tracks, resolve_audio_paths
from soundspace.space.embed.base import Embedder, load_embedder
from soundspace.space.search.index import SearchIndex
from soundspace.space.search.query import (
    embed_audio_query,
    embed_text_query,
    preprocess_query,
    retrieve_top_k,
)

log = logging.getLogger(__name__)
console = Console()

_EMBEDDINGS_SUBDIR = "embeddings"
_DEFAULT_TOP_K = 10

_OVERVIEW = """\
Search the embedding index by audio, text, or an existing track.

Loads data/processed/embeddings/<model>.npz as the index, preprocesses the
query to match how the index was built (centering and L2 mirror the stored
flags), and returns the nearest tracks by cosine similarity. Each result shows
the resolved audio path as a clickable link where the terminal supports it.
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
    help="Index to search. Defaults to the active model in config.",
)
_TOPK_OPTION = typer.Option(
    _DEFAULT_TOP_K,
    "-k",
    "--top",
    min=1,
    metavar="K",
    help="Number of results to return.",
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


def _embedding_config(model: str | None) -> EmbeddingConfig:
    pipe = _pipeline_config()
    name = model or pipe.active
    if name not in pipe.embeddings:
        known = ", ".join(sorted(pipe.embeddings))
        raise typer.BadParameter(f"unknown model {name!r}; choices: {known}")
    return pipe.embeddings[name]


def _load_index(dataset: DatasetConfig, model: str) -> SearchIndex:
    path = dataset.processed_dir / _EMBEDDINGS_SUBDIR / f"{model}.npz"
    if not path.exists():
        raise typer.BadParameter(
            f"index not found: {path}; run 'soundspace pipeline embed -m {model}' first"
        )
    return SearchIndex.load(path)


def _load_labels(dataset: DatasetConfig, model: str) -> dict[str, str]:
    path = dataset.artifacts_dir / f"{model}_clusters.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"song_id": str})
    return dict(zip(frame["song_id"], frame["label"]))


def _embedder(config: EmbeddingConfig) -> Embedder:
    with console.status(f"Loading {config.name} ({config.model_id})..."):
        return load_embedder(config)


def _path_cell(path: Path | None) -> Text:
    if path is None:
        return Text("")
    display = str(path)
    try:
        uri = path.resolve().as_uri()
    except ValueError:
        return Text(display)
    return Text(display, style=f"link {uri}")


def _print_results(
    dataset: DatasetConfig,
    model: str,
    query_label: str,
    results: list[tuple[str, float]],
) -> None:
    tracks = read_tracks(dataset)
    paths = resolve_audio_paths(tracks, dataset.audio_dir)
    meta = tracks.set_index("song_id")
    labels = _load_labels(dataset, model)
    has_labels = bool(labels)

    table = Table(title=f"Top {len(results)} for {query_label}")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("song_id", no_wrap=True)
    table.add_column("Artist")
    table.add_column("Title")
    if has_labels:
        table.add_column("Region")
    table.add_column("Score", justify="right")
    table.add_column("Path", no_wrap=True)

    for rank, (track_id, score) in enumerate(results, 1):
        artist = title = "?"
        if track_id in meta.index:
            row = meta.loc[track_id]
            artist = str(row.get("artist", "?"))
            title = str(row.get("title", "?"))

        cells: list[object] = [str(rank), track_id, Text(artist), Text(title)]
        if has_labels:
            cells.append(Text(labels.get(track_id, "")))
        cells.append(f"{score:.4f}")
        cells.append(_path_cell(paths.get(track_id)))
        table.add_row(*cells)
    console.print(table)


@app.command(help="Retrieve tracks similar to a query audio file.")
def audio(
    path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Audio file to query with."
    ),
    model: str | None = _MODEL_OPTION,
    top: int = _TOPK_OPTION,
) -> None:
    dataset = _config()
    config = _embedding_config(model)
    index = _load_index(dataset, config.name)
    embedder = _embedder(config)
    raw = embed_audio_query(path, embedder)
    query = preprocess_query(raw, index)
    results = retrieve_top_k(query, index, k=top)
    _print_results(dataset, config.name, path.name, results)


@app.command(help="Retrieve tracks matching a text description.")
def text(
    query: str = typer.Argument(
        ..., help="Text description, e.g. 'melancholic piano'."
    ),
    model: str | None = _MODEL_OPTION,
    top: int = _TOPK_OPTION,
) -> None:
    dataset = _config()
    config = _embedding_config(model)
    index = _load_index(dataset, config.name)
    embedder = _embedder(config)
    raw = embed_text_query(query, embedder)
    vector = preprocess_query(raw, index)
    results = retrieve_top_k(vector, index, k=top)
    _print_results(dataset, config.name, f'"{query}"', results)


@app.command(help="Retrieve tracks similar to an existing track (more like this).")
def track(
    song_id: str = typer.Argument(..., help="song_id of a track already in the index."),
    model: str | None = _MODEL_OPTION,
    top: int = _TOPK_OPTION,
) -> None:
    dataset = _config()
    name = model or _pipeline_config().active
    index = _load_index(dataset, name)
    pos = index.index_of(song_id)
    if pos is None:
        raise typer.BadParameter(f"track {song_id!r} not in index {name!r}")
    results = retrieve_top_k(index.embeddings[pos], index, k=top, exclude=[song_id])
    _print_results(dataset, name, f"track {song_id}", results)
