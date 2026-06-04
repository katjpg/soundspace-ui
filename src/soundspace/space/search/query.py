from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from soundspace.space.embed.base import Embedder
from soundspace.space.search.index import SearchIndex

_L2_EPS = 1e-12


@contextlib.contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stderr.flush()
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def preprocess_query(embedding: np.ndarray, index: SearchIndex) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if index.center:
        vec = vec - index.mean
    norm = float(np.linalg.norm(vec))
    if norm <= _L2_EPS:
        raise ValueError("zero-norm query after preprocessing")
    return (vec / norm).astype(np.float32)


def embed_audio_query(path: str | Path, embedder: Embedder) -> np.ndarray:
    import librosa

    with _suppress_native_stderr():
        waveform, _ = librosa.load(path, sr=embedder.sample_rate, mono=True)
    return embedder.embed_audio([np.asarray(waveform, dtype=np.float32)])[0]


def embed_text_query(text: str, embedder: Embedder) -> np.ndarray:
    return embedder.embed_text([text])[0]


def retrieve_top_k(
    query: np.ndarray,
    index: SearchIndex,
    k: int = 10,
    *,
    exclude: Sequence[str] = (),
) -> list[tuple[str, float]]:
    scores = index.embeddings @ np.asarray(query, dtype=np.float32)

    if exclude:
        drop = {str(t) for t in exclude}
        mask = np.array([str(t) in drop for t in index.track_ids])
        scores = scores.copy()
        scores[mask] = -np.inf

    k = min(k, len(scores))
    candidate_idx = np.argpartition(scores, -k)[-k:]
    candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
    return [(str(index.track_ids[i]), float(scores[i])) for i in candidate_idx]
