from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from soundspace.config.client import SpotifySettings
from soundspace.config.env import ENV_FILE, find_root
from soundspace.config.llm import LLMSettings

_CONFIG_FILE = Path("configs/config.yaml")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    root: Path = Field(default_factory=find_root)
    config_file: Path = _CONFIG_FILE
    llm: LLMSettings = Field(default_factory=LLMSettings)
    spotify: SpotifySettings = Field(default_factory=SpotifySettings)

    def config(self) -> dict[str, Any]:
        path = self.resolve_path(self.config_file)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        return yaml.safe_load(path.read_text())

    def resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
