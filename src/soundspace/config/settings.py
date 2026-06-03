from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

_CONFIG_FILE = Path("configs/config.yaml")


def _find_root() -> Path:
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / _CONFIG_FILE).is_file():
            return directory
    return Path.cwd()


class Settings(BaseSettings):
    root: Path = Field(default_factory=_find_root)
    config_file: Path = _CONFIG_FILE

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
