from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 44100


def load_audio(audio_path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError(f"empty audio: {audio_path}")

    return audio
