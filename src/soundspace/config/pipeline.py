from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DEFAULT_BATCH_SIZE = 8
_DEFAULT_MAX_DURATION = 30.0

DTypeName = Literal["float32", "float16", "bfloat16"]
EmbeddingProvider = Literal["huggingface"]
SymmetrizeMode = Literal["max", "min", "mean"]


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    provider: EmbeddingProvider = "huggingface"
    model_id: str
    device: str | None = None
    dtype: DTypeName = "float32"
    batch_size: int = Field(default=_DEFAULT_BATCH_SIZE, gt=0)
    center: bool = True
    normalize: bool = True
    max_duration: float = Field(default=_DEFAULT_MAX_DURATION, gt=0.0)


class KNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=10, gt=0)
    symmetrize: bool = True
    symmetrize_mode: SymmetrizeMode = "max"


class LeidenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: float = Field(default=1.0, gt=0.0)
    seed: int | None = 42
    n_iterations: int = -1
    use_weights: bool = True


class UMAPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_neighbors: int = Field(default=15, gt=1)
    min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    metric: Literal["cosine"] = "cosine"
    seed: int | None = 42


class LabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: tuple[str, ...] = ("mood_all",)
    top_n: int = Field(default=3, gt=0)


class ClusterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knn: KNNConfig = Field(default_factory=KNNConfig)
    leiden: LeidenConfig = Field(default_factory=LeidenConfig)
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: str
    embeddings: dict[str, EmbeddingConfig] = Field(min_length=1)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)

    @model_validator(mode="after")
    def _check_active(self) -> PipelineConfig:
        if self.active not in self.embeddings:
            known = ", ".join(sorted(self.embeddings)) or "<none>"
            raise ValueError(
                f"active embedding {self.active!r} not among configured models ({known})"
            )
        return self

    @property
    def embedding(self) -> EmbeddingConfig:
        return self.embeddings[self.active]


def load_pipeline_config(raw: dict[str, Any]) -> PipelineConfig:
    embeddings = raw.get("embeddings")
    if not isinstance(embeddings, dict):
        raise ValueError("config must define an 'embeddings' section")

    active = embeddings.get("active")
    if not isinstance(active, str) or not active:
        raise ValueError("config 'embeddings' must define an 'active' model name")

    models = embeddings.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("config 'embeddings' must define a non-empty 'models' map")

    named = {name: {**spec, "name": name} for name, spec in models.items()}
    pipeline = raw.get("pipeline") or {}
    return PipelineConfig.model_validate(
        {"active": active, "embeddings": named, "cluster": pipeline}
    )
