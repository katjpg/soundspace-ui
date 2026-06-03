from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from muq import MuQMuLan

from soundspace.config.pipeline import EmbeddingConfig
from soundspace.space.embed.runtime import check_rows, resolve_device, resolve_dtype

_MUQ_SAMPLE_RATE = 24000
_MUQ_DIM = 512


def _pad_or_trim(samples: np.ndarray, length: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size >= length:
        return samples[:length]
    return np.pad(samples, (0, length - samples.size))


@dataclass(frozen=True, slots=True)
class MuqMulanEmbedder:
    model: Any
    device: str
    max_samples: int
    sample_rate: int = _MUQ_SAMPLE_RATE
    dim: int = _MUQ_DIM

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> MuqMulanEmbedder:
        device = resolve_device(config.device)
        dtype = resolve_dtype(config.dtype, device)
        model = MuQMuLan.from_pretrained(config.model_id)
        model = model.to(device).eval()
        if dtype != torch.float32:
            model = model.to(dtype=dtype)
        return cls(
            model=model,
            device=device,
            max_samples=int(_MUQ_SAMPLE_RATE * config.max_duration),
        )

    @torch.no_grad()
    def embed_audio(self, audio: Sequence[np.ndarray]) -> np.ndarray:
        if len(audio) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        samples_batch = [_pad_or_trim(samples, self.max_samples) for samples in audio]
        batch = torch.from_numpy(np.stack(samples_batch)).to(self.device)
        embeddings = self.model(wavs=batch).float().cpu().numpy()
        return check_rows(embeddings, len(audio))

    @torch.no_grad()
    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        embeddings = self.model(texts=list(texts)).float().cpu().numpy()
        return check_rows(embeddings, len(texts))
