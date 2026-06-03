import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from soundspace.audio.extract import FeatureSet, extract as extract_features
from soundspace.audio.feature import ALL_FEATURE_GROUPS, FeatureGroup
from soundspace.config.dataset import DatasetConfig, load_dataset_config
from soundspace.config.pipeline import (
    EmbeddingConfig,
    PipelineConfig,
    load_pipeline_config,
)
from soundspace.config.settings import get_settings
from soundspace.space.embed.base import Embedder, load_embedder
from soundspace.space.pipeline.embed import EmbedResult, embed_dataset

log = logging.getLogger(__name__)
console = Console()

_MAX_FAILURE_LINES = 10
_PROFILE_PREVIEW_DIMS = 8
_PROFILE_SEED = 42

_OVERVIEW = """\
Commands for derived dataset artifacts.

These commands read the active dataset from config, compute pipeline artifacts,
and write them under data/.
"""


_EXTRACT_HELP = """\
Extract acoustic features from the cleaned dataset.

Reads data/processed/<dataset>.csv, computes rhythm, spectral, and tonal
features, and writes one feature row per track. Existing rows are skipped unless
--force is set.

Examples:

  soundspace pipeline extract

  soundspace pipeline extract -j 8 --features rhythm --features tonal

  soundspace pipeline extract --force

Output:

  data/processed/<dataset>_audio_features.csv
"""


_EMBED_HELP = """\
Embed dataset audio into dense vectors.

Reads data/processed/<dataset>.csv, loads each track's audio, encodes it with
the active embedding model, and writes one embedding matrix for the corpus.
Model and defaults come from config; the flags below override them per run.

Examples:

  soundspace pipeline embed

  soundspace pipeline embed -m clap -b 16

  soundspace pipeline embed -d cpu --no-center --no-normalize

Output:

  data/processed/embeddings/<model>.npz
"""

app = typer.Typer(
    help=_OVERVIEW,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
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


@app.command(help=_EXTRACT_HELP)
def extract(
    jobs: int | None = typer.Option(
        None,
        "-j",
        "--jobs",
        min=1,
        metavar="N",
        help="Worker processes to use. Defaults to CPU count.",
    ),
    feature_groups: list[FeatureGroup] = typer.Option(
        list(ALL_FEATURE_GROUPS),
        "--features",
        case_sensitive=False,
        help="Feature groups to extract; repeat to select multiple groups.",
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        "--overwrite",
        help="Recompute rows that already exist.",
    ),
) -> None:
    cfg = _config()
    out_path = cfg.processed_dir / f"{cfg.active}_audio_features.csv"

    existing = None if force else _read_existing_features(out_path)
    extracted = set(existing["song_id"].astype(str)) if existing is not None else set()
    pending = _pending_song_ids(cfg, extracted)

    if not pending:
        typer.echo(f"All tracks extracted: {out_path}")
        _print_profile(existing if existing is not None else pd.read_csv(out_path))
        return

    if extracted:
        typer.echo(f"Resuming: {len(extracted)} done, {len(pending)} pending")

    result = extract_features(
        cfg,
        feature_groups=tuple(feature_groups),
        num_workers=jobs,
        song_ids=pending,
    )

    if len(result) == 0 and existing is None:
        typer.echo("No tracks extracted; every track failed.")
        _echo_failures(result)
        raise typer.Exit(code=1)

    merged = _merge_features(existing, result.to_dataframe())
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    _echo_failures(result)
    typer.echo(f"Audio features: {out_path}")
    _print_profile(merged)


def _read_existing_features(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    table = pd.read_csv(path, dtype={"song_id": str})
    if "song_id" not in table.columns:
        raise ValueError(f"{path}: missing song_id column")
    return table


def _pending_song_ids(cfg: DatasetConfig, extracted_ids: set[str]) -> list[str]:
    table = pd.read_csv(cfg.processed_dir / f"{cfg.active}.csv", dtype={"song_id": str})
    ids = [str(song_id) for song_id in table["song_id"].tolist()]
    return [song_id for song_id in ids if song_id not in extracted_ids]


def _merge_features(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new
    combined = pd.concat([existing, new], ignore_index=True)
    return combined.drop_duplicates(subset="song_id", keep="last").reset_index(
        drop=True
    )


def _echo_failures(result: FeatureSet) -> None:
    if not result.failures:
        return
    typer.echo(f"Failed tracks: {len(result.failures)}")
    for failure in result.failures[:_MAX_FAILURE_LINES]:
        typer.echo(f"  {failure.song_id}: {failure.error}")


def _print_profile(table: pd.DataFrame) -> None:
    rows = len(table)
    profile = Table(title=f"{rows} rows x {len(table.columns)} columns")
    profile.add_column("Column", style="cyan", no_wrap=True)
    profile.add_column("Type")
    profile.add_column("Null %", justify="right")
    profile.add_column("Distinct", justify="right")
    profile.add_column("Min", justify="right")
    profile.add_column("Max", justify="right")

    for column in table.columns:
        series = table[column]
        null_pct = f"{series.isna().mean() * 100:.1f}" if rows else "0.0"
        distinct = str(series.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            minimum = f"{valid.min():.3g}" if not valid.empty else "-"
            maximum = f"{valid.max():.3g}" if not valid.empty else "-"
        else:
            minimum = maximum = "-"
        profile.add_row(column, str(series.dtype), null_pct, distinct, minimum, maximum)

    console.print(profile)


def _resolve_embedding_config(
    model: str | None,
    batch_size: int | None,
    device: str | None,
    center: bool | None,
    normalize: bool | None,
) -> EmbeddingConfig:
    pipe = _pipeline_config()
    name = model or pipe.active
    if name not in pipe.embeddings:
        known = ", ".join(sorted(pipe.embeddings))
        raise typer.BadParameter(f"unknown model {name!r}; choices: {known}")
    updates: dict[str, object] = {}
    if batch_size is not None:
        updates["batch_size"] = batch_size
    if device is not None:
        updates["device"] = device
    if center is not None:
        updates["center"] = center
    if normalize is not None:
        updates["normalize"] = normalize
    return pipe.embeddings[name].model_copy(update=updates)


def _load_embedder(config: EmbeddingConfig) -> Embedder:
    with console.status(f"Loading {config.name} ({config.model_id})..."):
        return load_embedder(config)


@app.command(help=_EMBED_HELP)
def embed(
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        metavar="NAME",
        help="Embedding model to use. Defaults to the active model in config.",
    ),
    batch_size: int | None = typer.Option(
        None,
        "-b",
        "--batch-size",
        min=1,
        metavar="N",
        help="Tracks per batch. Defaults to the model's configured batch size.",
    ),
    device: str | None = typer.Option(
        None,
        "-d",
        "--device",
        metavar="DEVICE",
        help="Compute device (cpu, cuda, mps). Defaults to auto-detect.",
    ),
    center: bool | None = typer.Option(
        None,
        "--center/--no-center",
        help="Subtract the corpus mean before normalizing. Defaults to config.",
    ),
    normalize: bool | None = typer.Option(
        None,
        "--normalize/--no-normalize",
        help="L2-normalize embeddings. Defaults to config.",
    ),
) -> None:
    dataset = _config()
    embedding = _resolve_embedding_config(model, batch_size, device, center, normalize)
    embedder = _load_embedder(embedding)
    result = embed_dataset(dataset, embedding, embedder=embedder)
    _echo_embed_failures(result)
    typer.echo(f"Embeddings: {result.path}")
    _print_embed_summary(embedding, result)
    _print_embed_profile(result)


def _echo_embed_failures(result: EmbedResult) -> None:
    if not result.failures:
        return
    typer.echo(f"Failed tracks: {len(result.failures)}")
    for failure in result.failures[:_MAX_FAILURE_LINES]:
        typer.echo(f"  {failure.song_id}: {failure.error}")


def _print_embed_summary(config: EmbeddingConfig, result: EmbedResult) -> None:
    table = Table(title=f"{config.name}: {len(result)} tracks x {result.dim} dims")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("name", config.name)
    table.add_row("model_id", config.model_id)
    table.add_row("device", config.device or "auto")
    table.add_row("batch_size", str(config.batch_size))
    table.add_row("center", str(config.center))
    table.add_row("normalize", str(config.normalize))
    table.add_row("tracks", str(len(result)))
    table.add_row("failures", str(len(result.failures)))
    console.print(table)


def _print_embed_profile(result: EmbedResult) -> None:
    data = np.load(result.path, allow_pickle=False)
    embeddings = data["embeddings"]
    mean = data["mean"]
    norms = np.linalg.norm(embeddings, axis=1)

    rng = np.random.default_rng(_PROFILE_SEED)
    idx = int(rng.integers(0, embeddings.shape[0]))
    preview = ", ".join(f"{v:+.3f}" for v in embeddings[idx, :_PROFILE_PREVIEW_DIMS])

    table = Table(title="Embedding profile")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("shape", f"{embeddings.shape[0]} x {embeddings.shape[1]}")
    table.add_row("dtype", str(embeddings.dtype))
    table.add_row("l2 norm mean", f"{float(norms.mean()):.4f}")
    table.add_row("l2 norm std", f"{float(norms.std()):.4f}")
    table.add_row(
        "value range", f"[{float(embeddings.min()):.3f}, {float(embeddings.max()):.3f}]"
    )
    table.add_row("mean-vector norm", f"{float(np.linalg.norm(mean)):.4f}")
    table.add_row(f"row[{idx}][:{_PROFILE_PREVIEW_DIMS}]", preview)
    console.print(table)
