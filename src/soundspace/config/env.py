from pathlib import Path

_CONFIG_FILE = Path("configs/config.yaml")


def find_root() -> Path:
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (directory / _CONFIG_FILE).is_file():
            return directory
    return Path.cwd()


ROOT = find_root()
ENV_FILE = ROOT / ".env"
