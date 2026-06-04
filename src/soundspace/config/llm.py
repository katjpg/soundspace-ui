from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from soundspace.config.env import ENV_FILE

_GEMINI_MODEL = "gemini-3-flash-preview"
_OPENROUTER_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_", env_file=ENV_FILE, extra="ignore"
    )

    api_key: SecretStr | None = None
    model: str = _GEMINI_MODEL
    max_output_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 40
    thinking_level: str = "minimal"

    @property
    def configured(self) -> bool:
        return self.api_key is not None


class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_", env_file=ENV_FILE, extra="ignore"
    )

    api_key: SecretStr | None = None
    model: str = _OPENROUTER_MODEL
    base_url: str = _OPENROUTER_BASE_URL
    max_tokens: int = 8192
    temperature: float = 0.3
    top_p: float = 0.95
    reasoning_enabled: bool = True
    referer: str | None = None
    title: str | None = None

    @property
    def configured(self) -> bool:
        return self.api_key is not None


class HuggingFaceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HF_", env_file=ENV_FILE, extra="ignore"
    )

    token: SecretStr | None = None

    @property
    def configured(self) -> bool:
        return self.token is not None


class RerankConfig(BaseModel):
    provider: str = "gemini"
    song_results: int = 5
    playlist_results: int = 20
    recent_seeds: int = 1
    top_seeds: int = 5
    timeout: float = 60.0


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
