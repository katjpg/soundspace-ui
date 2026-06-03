import logging

import typer

from soundspace.cli import dataset

app = typer.Typer(
    help="Soundspace CLI: command-line interface for dataset and pipeline workflows.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(dataset.app, name="dataset")


@app.callback()
def configure_logging(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show debug logs."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
