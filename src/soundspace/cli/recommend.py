import asyncio
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from soundspace.client.deezer import DeezerClient
from soundspace.config.dataset import DatasetConfig
from soundspace.config.pipeline import EmbeddingConfig
from soundspace.config.settings import Settings, get_settings
from soundspace.llm.providers.base import LLMProvider
from soundspace.schemas.recommend import RecommendRequest
from soundspace.services import runtime
from soundspace.services.recommend import (
    Recommendation,
    RecommendError,
    RecommendResult,
    RecommendService,
    SeedResolution,
)
from soundspace.services.runtime import ConfigError
from soundspace.space.embed.base import Embedder, load_embedder

_DEFAULT_DAEMON_URL = "http://127.0.0.1:8000"
_DAEMON_TIMEOUT = 300.0

log = logging.getLogger(__name__)
console = Console()

_TIME_RANGES = ("short_term", "medium_term", "long_term")

_OVERVIEW = """\
Recommend tracks from the SoundSpace index using LLM reranking.

Unlike 'search' (a local cosine query), 'recommend' brings in external or user
context (text, an audio file, a Spotify track or playlist, your top or recently
played tracks), retrieves candidate tracks, and reranks them with an LLM that
weighs mood, theme, style, and valence/arousal against your request.

Audio-only seeds skip the LLM and return nearest neighbours by similarity.
Spotify inputs are resolved to playable audio via Deezer 30s previews.
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
    help="Embedding index to use. Defaults to the active model.",
)
_PROVIDER_OPTION = typer.Option(
    None,
    "-p",
    "--provider",
    metavar="NAME",
    help="LLM provider: gemini or openrouter. Defaults to config.",
)
_TEXT_OPTION = typer.Option(
    None,
    "--text",
    metavar="TEXT",
    help="Free-text request, e.g. 'late-night reflective rock'.",
)
_AUDIO_OPTION = typer.Option(
    None, "--audio", metavar="PATH", help="Local audio file to use as the seed."
)
_SPOTIFY_TRACK_OPTION = typer.Option(
    None, "--spotify-track", metavar="URL", help="Spotify track URL or id to seed from."
)
_SPOTIFY_PLAYLIST_OPTION = typer.Option(
    None,
    "--spotify-playlist",
    metavar="URL",
    help="Spotify playlist URL or id to seed from.",
)
_TOP_OPTION = typer.Option(False, "--top", help="Seed from your Spotify top tracks.")
_RECENT_OPTION = typer.Option(
    False, "--recent", help="Seed from your most recently played Spotify tracks."
)
_TIME_RANGE_OPTION = typer.Option(
    "medium_term",
    "--time-range",
    help="Top-tracks window: short_term, medium_term, long_term.",
)
_QUIET_OPTION = typer.Option(
    False, "-q", "--quiet", help="Suppress step-by-step progress output."
)
_DAEMON_OPTION = typer.Option(
    False,
    "--daemon",
    help="Route the request through a running 'soundspace serve' daemon.",
)
_DAEMON_URL_OPTION = typer.Option(
    None,
    "--daemon-url",
    metavar="URL",
    help=(
        f"Daemon base URL. Defaults to $SOUNDSPACE_DAEMON_URL or {_DEFAULT_DAEMON_URL}."
    ),
)


class _LazyEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._inner: Embedder | None = None

    def _load(self) -> Embedder:
        if self._inner is None:
            with console.status(
                f"Loading {self._config.name} ({self._config.model_id})..."
            ):
                self._inner = load_embedder(self._config)
        return self._inner

    @property
    def sample_rate(self) -> int:
        return self._load().sample_rate

    @property
    def dim(self) -> int:
        return self._load().dim

    def embed_audio(self, audio):  # type: ignore[no-untyped-def]
        return self._load().embed_audio(audio)

    def embed_text(self, texts):  # type: ignore[no-untyped-def]
        return self._load().embed_text(texts)


_SEED_SOURCES = ("audio", "spotify_track", "spotify_playlist", "top", "recent")


def _resolve_source(
    *,
    text: str | None,
    audio: str | None,
    spotify_track: str | None,
    spotify_playlist: str | None,
    top: bool,
    recent: bool,
) -> str:
    seeds = {
        "audio": audio,
        "spotify_track": spotify_track,
        "spotify_playlist": spotify_playlist,
        "top": top,
        "recent": recent,
    }
    chosen = [name for name, value in seeds.items() if value]
    if len(chosen) > 1:
        raise typer.BadParameter(
            "provide at most one seed source: "
            + ", ".join(f"--{n.replace('_', '-')}" for n in _SEED_SOURCES)
        )
    if chosen:
        return chosen[0]
    if text:
        return "text"
    raise typer.BadParameter(
        "provide a seed: --text, "
        + ", ".join(f"--{n.replace('_', '-')}" for n in _SEED_SOURCES)
    )


def _progress(message: str) -> None:
    console.print(f"[dim]>[/dim] {message}")


def _path_cell(path: Path | None) -> Text:
    if path is None:
        return Text("")
    display = str(path)
    try:
        uri = path.resolve().as_uri()
    except ValueError:
        return Text(display)
    return Text(display, style=f"link {uri}")


def _print_recommendations(result: RecommendResult, query_label: str) -> None:
    if not result.recommendations:
        console.print("[yellow]No recommendations produced.[/yellow]")
    else:
        table = Table(title=f"Recommendations for {query_label}")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("song_id", no_wrap=True)
        table.add_column("Artist")
        table.add_column("Title")
        table.add_column("Region")
        table.add_column("Fit", justify="right")
        table.add_column("Why")
        table.add_column("Path", no_wrap=True)
        for i, rec in enumerate(result.recommendations, 1):
            table.add_row(
                str(i),
                rec.song_id,
                rec.artist,
                rec.title,
                rec.region,
                f"{rec.fit:.2f}",
                rec.reason,
                _path_cell(rec.path),
            )
        console.print(table)
    if result.resolved_seeds:
        console.print(f"[green]Seeds resolved:[/green] {len(result.resolved_seeds)}")
    if result.unresolved_seeds:
        console.print(
            f"[yellow]Unresolved seeds ({len(result.unresolved_seeds)}):[/yellow]"
        )
        for seed in result.unresolved_seeds:
            console.print(f"  {seed.artist} - {seed.title}: {seed.reason}")


async def _run(
    settings: Settings,
    dataset: DatasetConfig,
    config: EmbeddingConfig,
    provider: LLMProvider,
    *,
    source: str,
    text: str | None,
    audio: str | None,
    spotify_track: str | None,
    spotify_playlist: str | None,
    time_range: str,
    n: int,
    quiet: bool,
) -> RecommendResult:
    embedder = _LazyEmbedder(config)
    progress = None if quiet else _progress
    async with AsyncExitStack() as stack:
        deezer = await stack.enter_async_context(DeezerClient())
        spotify_client = runtime.spotify_client(settings)
        spotify = (
            await stack.enter_async_context(spotify_client)
            if spotify_client is not None
            else None
        )
        svc = RecommendService(
            dataset,
            model=config.name,
            embedder=embedder,
            provider=provider,
            config=settings.llm.rerank,
            spotify=spotify,
            deezer=deezer,
            progress=progress,
        )
        if source == "text":
            return await svc.from_text(text or "", n=n)
        if source == "audio":
            return await svc.from_audio(audio or "", query=text, n=n)
        if source == "spotify_track":
            return await svc.from_spotify_track(spotify_track or "", query=text, n=n)
        if source == "spotify_playlist":
            return await svc.from_spotify_playlist(
                spotify_playlist or "", query=text, n=n
            )
        if source == "top":
            return await svc.from_top(time_range=time_range, query=text, n=n)
        return await svc.from_recent(query=text, n=n)


def _daemon_base_url(url: str | None) -> str:
    chosen = url or os.environ.get("SOUNDSPACE_DAEMON_URL") or _DEFAULT_DAEMON_URL
    return chosen.rstrip("/")


def _daemon_detail(response: httpx.Response, fallback: str) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return str(detail) if detail else fallback


def _result_from_json(data: dict[str, object]) -> RecommendResult:
    recommendations = [
        Recommendation(
            song_id=item["song_id"],
            artist=item["artist"],
            title=item["title"],
            region=item["region"],
            fit=item["fit"],
            reason=item["reason"],
            path=Path(item["path"]) if item.get("path") else None,
        )
        for item in data.get("recommendations", [])
    ]
    resolved = [SeedResolution(**s) for s in data.get("resolved_seeds", [])]
    unresolved = [SeedResolution(**s) for s in data.get("unresolved_seeds", [])]
    return RecommendResult(
        recommendations=recommendations,
        resolved_seeds=resolved,
        unresolved_seeds=unresolved,
        candidate_count=int(data.get("candidate_count", 0)),
    )


def _post_recommend(
    *,
    base_url: str,
    mode: str,
    source: str,
    n: int,
    text: str | None,
    audio: str | None,
    spotify_track: str | None,
    spotify_playlist: str | None,
    top: bool,
    recent: bool,
    time_range: str,
    model: str | None,
    provider: str | None,
) -> RecommendResult:
    request = RecommendRequest(
        mode=mode,
        text=text,
        spotify_track=spotify_track,
        spotify_playlist=spotify_playlist,
        top=top,
        recent=recent,
        time_range=time_range,
        n=n,
        model=model,
        provider=provider,
    )
    handle = None
    files = None
    if source == "audio":
        path = Path(audio or "")
        if not path.is_file():
            raise typer.BadParameter(f"audio file not found: {path}")
        handle = path.open("rb")
        files = {"audio": (path.name, handle, "application/octet-stream")}
    try:
        response = httpx.post(
            f"{base_url}/recommend",
            data={"payload": request.model_dump_json()},
            files=files,
            timeout=_DAEMON_TIMEOUT,
        )
    except httpx.ConnectError as exc:
        raise typer.BadParameter(
            f"daemon not reachable at {base_url}; start it with 'soundspace serve'"
        ) from exc
    except httpx.HTTPError as exc:
        raise typer.BadParameter(f"daemon request failed: {exc}") from exc
    finally:
        if handle is not None:
            handle.close()
    if response.status_code == 503:
        raise typer.BadParameter(
            _daemon_detail(response, "daemon is warming up; retry shortly")
        )
    if response.status_code != 200:
        raise typer.BadParameter(
            _daemon_detail(response, f"daemon returned HTTP {response.status_code}")
        )
    return _result_from_json(response.json())


def _recommend(
    *,
    mode: str,
    default_n: int,
    text: str | None,
    audio: str | None,
    spotify_track: str | None,
    spotify_playlist: str | None,
    top: bool,
    recent: bool,
    time_range: str,
    n: int | None,
    model: str | None,
    provider: str | None,
    quiet: bool,
    daemon: bool,
    daemon_url: str | None,
    query_label_prefix: str = "",
) -> None:
    if time_range not in _TIME_RANGES:
        raise typer.BadParameter(f"time-range must be one of {', '.join(_TIME_RANGES)}")
    source = _resolve_source(
        text=text,
        audio=audio,
        spotify_track=spotify_track,
        spotify_playlist=spotify_playlist,
        top=top,
        recent=recent,
    )
    count = n if n is not None else default_n
    label = source.replace("_", " ")
    if daemon:
        result = _post_recommend(
            base_url=_daemon_base_url(daemon_url),
            mode=mode,
            source=source,
            n=count,
            text=text,
            audio=audio,
            spotify_track=spotify_track,
            spotify_playlist=spotify_playlist,
            top=top,
            recent=recent,
            time_range=time_range,
            model=model,
            provider=provider,
        )
        _print_recommendations(result, f"{query_label_prefix}{label}")
        return
    settings = get_settings()
    try:
        dataset = runtime.dataset_config()
        config = runtime.embedding_config(model)
        llm = runtime.provider(settings, provider)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        result = asyncio.run(
            _run(
                settings,
                dataset,
                config,
                llm,
                source=source,
                text=text,
                audio=audio,
                spotify_track=spotify_track,
                spotify_playlist=spotify_playlist,
                time_range=time_range,
                n=count,
                quiet=quiet,
            )
        )
    except RecommendError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_recommendations(result, f"{query_label_prefix}{label}")


@app.command(help="Recommend songs from a text, audio, or Spotify seed.")
def song(
    text: str | None = _TEXT_OPTION,
    audio: str | None = _AUDIO_OPTION,
    spotify_track: str | None = _SPOTIFY_TRACK_OPTION,
    spotify_playlist: str | None = _SPOTIFY_PLAYLIST_OPTION,
    top: bool = _TOP_OPTION,
    recent: bool = _RECENT_OPTION,
    time_range: str = _TIME_RANGE_OPTION,
    n: int | None = typer.Option(
        None,
        "-n",
        "--n",
        min=1,
        max=50,
        help="Number of songs to recommend. Defaults to config.",
    ),
    model: str | None = _MODEL_OPTION,
    provider: str | None = _PROVIDER_OPTION,
    quiet: bool = _QUIET_OPTION,
    daemon: bool = _DAEMON_OPTION,
    daemon_url: str | None = _DAEMON_URL_OPTION,
) -> None:
    _recommend(
        mode="song",
        default_n=get_settings().llm.rerank.song_results,
        text=text,
        audio=audio,
        spotify_track=spotify_track,
        spotify_playlist=spotify_playlist,
        top=top,
        recent=recent,
        time_range=time_range,
        n=n,
        model=model,
        provider=provider,
        quiet=quiet,
        daemon=daemon,
        daemon_url=daemon_url,
    )


@app.command(
    help="Recommend a playlist (up to 20 songs) from a text, audio, or Spotify seed."
)
def playlist(
    text: str | None = _TEXT_OPTION,
    audio: str | None = _AUDIO_OPTION,
    spotify_track: str | None = _SPOTIFY_TRACK_OPTION,
    spotify_playlist: str | None = _SPOTIFY_PLAYLIST_OPTION,
    top: bool = _TOP_OPTION,
    recent: bool = _RECENT_OPTION,
    time_range: str = _TIME_RANGE_OPTION,
    n: int | None = typer.Option(
        None,
        "-n",
        "--n",
        min=1,
        max=20,
        help="Songs in the playlist (max 20). Defaults to config.",
    ),
    model: str | None = _MODEL_OPTION,
    provider: str | None = _PROVIDER_OPTION,
    quiet: bool = _QUIET_OPTION,
    daemon: bool = _DAEMON_OPTION,
    daemon_url: str | None = _DAEMON_URL_OPTION,
) -> None:
    _recommend(
        mode="playlist",
        default_n=get_settings().llm.rerank.playlist_results,
        text=text,
        audio=audio,
        spotify_track=spotify_track,
        spotify_playlist=spotify_playlist,
        top=top,
        recent=recent,
        time_range=time_range,
        n=n,
        model=model,
        provider=provider,
        quiet=quiet,
        daemon=daemon,
        daemon_url=daemon_url,
        query_label_prefix="playlist: ",
    )
