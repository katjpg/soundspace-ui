from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from soundspace.audio.feature import load_audio
from soundspace.config.dataset import DatasetConfig
from soundspace.config.pipeline import EmbeddingConfig
from soundspace.dataset.tracks import read_tracks, resolve_audio_paths
from soundspace.space.embed.base import Embedder, load_embedder

_EMBEDDINGS_SUBDIR = "embeddings"


@dataclass(frozen=True, slots=True)
class EmbedFailure:
    song_id: str
    error: str


@dataclass(frozen=True, slots=True)
class EmbedResult:
    path: Path
    track_ids: list[str]
    dim: int
    failures: list[EmbedFailure]

    def __len__(self) -> int:
        return len(self.track_ids)


def embed_dataset(
    dataset: DatasetConfig,
    embedding: EmbeddingConfig,
    *,
    embedder: Embedder | None = None,
    progress: bool = True,
) -> EmbedResult:
    if embedder is None:
        embedder = load_embedder(embedding)
    table = read_tracks(dataset)
    audio_paths = resolve_audio_paths(table, dataset.audio_dir)
    if not audio_paths:
        raise ValueError(f"no tracks found for dataset {dataset.active!r}")

    items = list(audio_paths.items())
    n_batches = (len(items) + embedding.batch_size - 1) // embedding.batch_size
    batches: Iterator[list[tuple[str, Path]]] = _batched(items, embedding.batch_size)
    if progress:
        batches = tqdm(batches, total=n_batches, desc="Embedding")

    track_ids: list[str] = []
    embedding_blocks: list[np.ndarray] = []
    failures: list[EmbedFailure] = []

    for batch in batches:
        audio_batch: list[np.ndarray] = []
        batch_ids: list[str] = []
        for song_id, audio_path in batch:
            try:
                audio_batch.append(
                    load_audio(audio_path, sample_rate=embedder.sample_rate)
                )
                batch_ids.append(song_id)
            except Exception as exc:
                failures.append(EmbedFailure(song_id=song_id, error=str(exc)))
        if not audio_batch:
            continue
        embedding_blocks.append(embedder.embed_audio(audio_batch))
        track_ids.extend(batch_ids)

    if not embedding_blocks:
        raise RuntimeError("every track failed to embed")

    embeddings = np.vstack(embedding_blocks).astype(np.float32)
    embedding_mean = np.zeros(embeddings.shape[1], dtype=np.float32)
    if embedding.center:
        embedding_mean = embeddings.mean(axis=0).astype(np.float32)
        embeddings = (embeddings - embedding_mean).astype(np.float32)
    if embedding.normalize:
        embeddings = _l2_normalize(embeddings)

    out_path = dataset.processed_dir / _EMBEDDINGS_SUBDIR / f"{embedding.name}.npz"
    _save(out_path, track_ids, embeddings, embedding_mean, embedding, embedder)

    return EmbedResult(
        path=out_path,
        track_ids=track_ids,
        dim=embeddings.shape[1],
        failures=failures,
    )


def _batched(
    items: list[tuple[str, Path]],
    size: int,
) -> Iterator[list[tuple[str, Path]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if float(norms.min()) <= 0.0:
        raise ValueError("zero-norm vector; cannot L2-normalize")
    return (embeddings / norms).astype(np.float32)


def _save(
    path: Path,
    track_ids: list[str],
    embeddings: np.ndarray,
    mean: np.ndarray,
    config: EmbeddingConfig,
    embedder: Embedder,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        track_ids=np.asarray(track_ids),
        embeddings=embeddings.astype(np.float32),
        mean=mean.astype(np.float32),
        name=np.asarray(config.name),
        model_id=np.asarray(config.model_id),
        sample_rate=np.asarray(embedder.sample_rate),
        center=np.asarray(config.center),
        normalize=np.asarray(config.normalize),
    )
