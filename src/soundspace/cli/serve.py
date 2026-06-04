import typer
import uvicorn

from soundspace.app import create_app

app = typer.Typer(
    help="Run the SoundSpace recommendation daemon (warm model + index).",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback(invoke_without_command=True)
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        metavar="NAME",
        help="Embedding model to warm. Defaults to the active model.",
    ),
) -> None:
    application = create_app(model=model)
    uvicorn.run(application, host=host, port=port, log_level="info")
