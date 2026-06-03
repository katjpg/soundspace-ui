import copy
import logging
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from soundspace.audio.extract import FeatureSet, extract as extract_features
from soundspace.audio.feature import ALL_FEATURE_GROUPS, FeatureGroup
from soundspace.config.dataset import DatasetConfig, load_dataset_config
from soundspace.config.settings import get_settings

log = logging.getLogger(__name__)
console = Console()

_MAX_FAILURE_LINES = 10

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
