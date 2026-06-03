import numpy as np
import torch

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype != "float32" and device != "cuda":
        raise ValueError(f"{dtype} embeddings require cuda, not {device}")
    return DTYPES[dtype]


def check_rows(embeddings: np.ndarray, n_rows: int) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] != n_rows:
        raise ValueError(
            f"expected ({n_rows}, dim) embeddings, got shape {embeddings.shape}"
        )
    return embeddings
