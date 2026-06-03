import copy
import logging

import typer

from soundspace.config.dataset import DatasetConfig, load_dataset_config
from soundspace.config.settings import get_settings
from soundspace.dataset.download import download as download_dataset
from soundspace.dataset.filter import TAG_GROUP_COLUMNS, FilterConfig
from soundspace.dataset.labels import build_labels
from soundspace.dataset.preprocess import preprocess as preprocess_dataset
from soundspace.types import TagGroup

log = logging.getLogger(__name__)

LABELS_FILE = "labels.json"
_TAG_GROUPS: tuple[TagGroup, ...] = tuple(TAG_GROUP_COLUMNS)

_OVERVIEW = """\
Commands for the MERGE Audio Balanced dataset.

MERGE Audio Balanced contains 3,232 30-second audio samples labeled by
Russell quadrant, with AllMusic mood, genre, theme, and style tags and
per-track arousal-valence values. These commands download the archive,
clean and filter the metadata, and write processed tables under data/.
"""


_DOWNLOAD_HELP = """\
Download the MERGE Audio Balanced dataset.

Downloads the archive from Zenodo, verifies its checksum, extracts the
files, and prepares the dataset directory.

Example:

  soundspace dataset download

NOTE: This command is safe to run multiple times: if a valid archive is
present, or the dataset has already been extracted, it is reused instead of
downloaded again.
"""


_PREPROCESS_HELP = """\
Clean and filter the MERGE Audio Balanced metadata.

Merges metadata with arousal-valence values, standardizes tag fields, adds
genre_main, and drops tracks missing a required tag group or audio file.
Also writes the label vocabulary.

Examples:

  soundspace dataset preprocess

  soundspace dataset preprocess -r mood -r genre -n 100 --seed 42

Outputs:

  data/processed/<dataset>.csv
  data/processed/labels.json
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


@app.command(help=_DOWNLOAD_HELP)
def download() -> None:
    cfg = _config()
    path = download_dataset(cfg)
    typer.echo(f"Dataset ready: {path}")


@app.command(help=_PREPROCESS_HELP)
def preprocess(
    required_groups: list[str] = typer.Option(
        list(_TAG_GROUPS),
        "-r",
        "--require",
        metavar="GROUP",
        help=(
            "Keep only tracks that have these tag groups; repeat to add more. "
            f"Choices: {', '.join(_TAG_GROUPS)}."
        ),
    ),
    sample_size: int | None = typer.Option(
        None,
        "-n",
        "--sample",
        min=1,
        metavar="N",
        help="Keep a reproducible sample of N tracks after filtering.",
    ),
    seed: int = typer.Option(
        7,
        "--seed",
        metavar="SEED",
        help="Random seed used with --sample.",
    ),
) -> None:
    cfg = _config()
    groups = _parse_tag_groups(required_groups)

    config = FilterConfig(
        required_groups=groups,
        sample_size=sample_size,
        seed=seed,
    )

    table = preprocess_dataset(
        cfg.metadata_path,
        cfg.av_path,
        cfg.audio_dir,
        cfg.processed_dir / f"{cfg.active}.csv",
        config=config,
    )
    labels = build_labels(cfg.metadata_path, cfg.processed_dir / LABELS_FILE)

    typer.echo(f"Clean table: {table}")
    typer.echo(f"Labels: {labels}")


def _parse_tag_groups(groups: list[str]) -> tuple[TagGroup, ...]:
    unknown = sorted(set(groups) - set(_TAG_GROUPS))
    if unknown:
        raise typer.BadParameter(
            f"unknown tag group(s): {', '.join(unknown)}. "
            f"valid choices: {', '.join(_TAG_GROUPS)}."
        )
    return tuple(groups)  # type: ignore[return-value]
