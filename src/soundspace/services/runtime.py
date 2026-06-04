import copy

from soundspace.client.spotify import SpotifyClient
from soundspace.config.dataset import DatasetConfig, load_dataset_config
from soundspace.config.pipeline import (
    EmbeddingConfig,
    PipelineConfig,
    load_pipeline_config,
)
from soundspace.config.settings import Settings, get_settings
from soundspace.llm.providers.base import LLMProvider
from soundspace.llm.providers.gemini import GeminiParams, GeminiProvider
from soundspace.llm.providers.openrouter import OpenRouterParams, OpenRouterProvider


class ConfigError(ValueError): ...


def dataset_config() -> DatasetConfig:
    settings = get_settings()
    raw = copy.deepcopy(settings.config())
    raw["data"]["root"] = str(settings.resolve_path(raw["data"]["root"]))
    return load_dataset_config(raw)


def pipeline_config() -> PipelineConfig:
    return load_pipeline_config(copy.deepcopy(get_settings().config()))


def embedding_config(model: str | None) -> EmbeddingConfig:
    pipe = pipeline_config()
    name = model or pipe.active
    if name not in pipe.embeddings:
        known = ", ".join(sorted(pipe.embeddings))
        raise ConfigError(f"unknown model {name!r}; choices: {known}")
    return pipe.embeddings[name]


def provider(settings: Settings, name: str | None) -> LLMProvider:
    rerank = settings.llm.rerank
    choice = name or rerank.provider
    if choice == "gemini":
        cfg = settings.llm.gemini
        if not cfg.configured:
            raise ConfigError("Gemini not configured; set GEMINI_API_KEY")
        return GeminiProvider(
            GeminiParams(
                api_key=cfg.api_key.get_secret_value(),
                model=cfg.model,
                max_output_tokens=cfg.max_output_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                thinking_level=cfg.thinking_level,
                timeout=rerank.timeout,
            )
        )
    if choice == "openrouter":
        cfg = settings.llm.openrouter
        if not cfg.configured:
            raise ConfigError("OpenRouter not configured; set OPENROUTER_API_KEY")
        return OpenRouterProvider(
            OpenRouterParams(
                api_key=cfg.api_key.get_secret_value(),
                model=cfg.model,
                base_url=cfg.base_url,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                reasoning_enabled=cfg.reasoning_enabled,
                referer=cfg.referer,
                title=cfg.title,
                timeout=rerank.timeout,
            )
        )
    raise ConfigError(f"unknown provider {choice!r}; choices: gemini, openrouter")


def spotify_client(settings: Settings) -> SpotifyClient | None:
    cfg = settings.spotify
    if not cfg.configured:
        return None
    return SpotifyClient(
        client_id=cfg.client_id.get_secret_value(),
        client_secret=cfg.client_secret.get_secret_value(),
        refresh_token=cfg.refresh_token.get_secret_value(),
    )
