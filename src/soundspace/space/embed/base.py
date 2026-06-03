from collections.abc import Sequence
from typing import Protocol

import numpy as np

from soundspace.config.pipeline import EmbeddingConfig


class Embedder(Protocol):
    sample_rate: int
    dim: int

    def embed_audio(self, audio: Sequence[np.ndarray]) -> np.ndarray: ...

    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...


def load_embedder(config: EmbeddingConfig) -> Embedder:
    if config.name == "muq_mulan":
        from soundspace.space.embed.muq_mulan import MuqMulanEmbedder

        return MuqMulanEmbedder.from_config(config)
    if config.name == "clap":
        from soundspace.space.embed.clap import ClapEmbedder

        return ClapEmbedder.from_config(config)
    raise ValueError(f"unknown embedding model: {config.name!r}")
