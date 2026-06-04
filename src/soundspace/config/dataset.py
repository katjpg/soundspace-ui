from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

_MD5_PREFIX = "md5:"
_MD5_RE = re.compile(rf"^({_MD5_PREFIX})?[a-fA-F0-9]{{32}}$")


class AudioSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["zenodo"] = "zenodo"
    record_id: int = Field(gt=0)
    filename: str = Field(min_length=1)
    md5: str | None = None
    base_url: str = "https://zenodo.org"
    extract: bool = True

    @field_validator("md5")
    @classmethod
    def _check_md5(cls, value: str | None) -> str | None:
        if value is None:
            return None
        md5 = value.strip()
        if not _MD5_RE.fullmatch(md5):
            raise ValueError(
                "md5 must be 32 hex characters, optionally prefixed by 'md5:'"
            )
        return md5

    @property
    def expected_md5(self) -> str | None:
        if self.md5 is None:
            return None
        return self.md5.strip().lower().removeprefix(_MD5_PREFIX)


class AudioFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path = Path(".")
    pattern: str = "*.mp3"


class MetadataFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: Path
    av_values: Path


class SplitFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    train_file: Path
    val_file: Path
    test_file: Path


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: AudioSource
    audio: AudioFiles
    metadata: MetadataFiles
    split: SplitFiles


class DataPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path
    name: str = Field(min_length=1)
    raw: Path
    processed: Path
    artifacts: Path


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataPaths
    datasets: dict[str, DatasetEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_active(self) -> DatasetConfig:
        if self.active not in self.datasets:
            known = ", ".join(sorted(self.datasets)) or "<none>"
            raise ValueError(
                f"active dataset {self.active!r} not among configured "
                f"datasets ({known})"
            )
        return self

    @property
    def active(self) -> str:
        return self.data.name

    @property
    def entry(self) -> DatasetEntry:
        return self.datasets[self.active]

    @property
    def source(self) -> AudioSource:
        return self.entry.source

    @property
    def raw_dir(self) -> Path:
        return self.data.root / self.data.raw

    @property
    def processed_dir(self) -> Path:
        return self.data.root / self.data.processed

    @property
    def artifacts_dir(self) -> Path:
        return self.data.root / self.data.artifacts

    @property
    def dataset_dir(self) -> Path:
        return self.raw_dir / self.active

    @property
    def audio_dir(self) -> Path:
        return self.dataset_dir / self.entry.audio.dir

    @property
    def metadata_path(self) -> Path:
        return self.dataset_dir / self.entry.metadata.file

    @property
    def av_path(self) -> Path:
        return self.dataset_dir / self.entry.metadata.av_values


def load_dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    data = raw.get("data")
    datasets = raw.get("datasets")
    if not data or not datasets:
        raise ValueError("config must define 'data' and 'datasets' sections")
    return DatasetConfig.model_validate({"data": data, "datasets": datasets})
