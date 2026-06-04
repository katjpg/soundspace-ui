import logging

import typer

from soundspace.cli import dataset, pipeline
from soundspace.cli import eval as eval_commands
from soundspace.cli.logging import configure_runtime

configure_runtime()

app = typer.Typer(
    help="Soundspace CLI: command-line interface for dataset and pipeline workflows.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(dataset.app, name="dataset")
app.add_typer(pipeline.app, name="pipeline")
app.add_typer(eval_commands.app, name="eval")


@app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Show debug logs.",
    ),
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
