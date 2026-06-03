from dataclasses import dataclass

import essentia.standard as es
import librosa
import numpy as np

from .io import SAMPLE_RATE

_EPS = 1e-12
_MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11)
_MINOR_DEGREES = (0, 2, 3, 5, 7, 8, 10)

_HPCP_FRAME_SIZE = 4096
_HPCP_HOP_SIZE = 2048
_HPCP_SIZE = 12
_HPCP_MAX_PEAKS = 100
_HPCP_MIN_FREQUENCY = 20
_HPCP_MAX_FREQUENCY = 5000
_HPCP_MAGNITUDE_THRESHOLD = 1e-5

_KEY_INDEX = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


@dataclass(frozen=True, slots=True)
class TonalFeatures:
    chroma_entropy: float
    major_alignment: float
    minor_alignment: float
    hpcp_0: float
    hpcp_1: float
    hpcp_2: float
    hpcp_3: float
    hpcp_4: float
    hpcp_5: float
    hpcp_6: float
    hpcp_7: float
    hpcp_8: float
    hpcp_9: float
    hpcp_10: float
    hpcp_11: float
    hpcp_entropy: float
    hpcp_std: float
    hpcp_max: float
    hpcp_temporal_std: float
    key_strength: float
    is_minor: float
    key_cos: float
    key_sin: float


@dataclass(frozen=True, slots=True)
class _HpcpStats:
    bins: tuple[float, ...]
    entropy: float
    std: float
    max: float
    temporal_std: float


@dataclass(frozen=True, slots=True)
class _KeyStats:
    strength: float
    is_minor: float
    cos: float
    sin: float


def compute_tonal(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> TonalFeatures:
    audio = np.asarray(audio, dtype=np.float32)

    chroma_mean = _chroma_mean(audio, sample_rate)
    hpcp = _hpcp(audio, sample_rate)
    key = _key(audio, sample_rate)

    return TonalFeatures(
        chroma_entropy=_entropy(chroma_mean),
        major_alignment=_alignment(chroma_mean, _MAJOR_DEGREES),
        minor_alignment=_alignment(chroma_mean, _MINOR_DEGREES),
        hpcp_0=hpcp.bins[0],
        hpcp_1=hpcp.bins[1],
        hpcp_2=hpcp.bins[2],
        hpcp_3=hpcp.bins[3],
        hpcp_4=hpcp.bins[4],
        hpcp_5=hpcp.bins[5],
        hpcp_6=hpcp.bins[6],
        hpcp_7=hpcp.bins[7],
        hpcp_8=hpcp.bins[8],
        hpcp_9=hpcp.bins[9],
        hpcp_10=hpcp.bins[10],
        hpcp_11=hpcp.bins[11],
        hpcp_entropy=hpcp.entropy,
        hpcp_std=hpcp.std,
        hpcp_max=hpcp.max,
        hpcp_temporal_std=hpcp.temporal_std,
        key_strength=key.strength,
        is_minor=key.is_minor,
        key_cos=key.cos,
        key_sin=key.sin,
    )


def _chroma_mean(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma = np.asarray(chroma, dtype=np.float32)
    if chroma.size == 0:
        return np.zeros(12, dtype=np.float32)
    mean_vec = np.asarray(np.mean(chroma, axis=1), dtype=np.float32).reshape(-1)
    if mean_vec.size != 12:
        raise ValueError(f"unexpected chroma size: {mean_vec.size}")
    return mean_vec


def _entropy(values: np.ndarray) -> float:
    weights = np.asarray(values, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    probabilities = np.clip(weights / total, _EPS, 1.0)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _alignment(chroma_mean: np.ndarray, degrees: tuple[int, ...]) -> float:
    weights = np.asarray(chroma_mean, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    return float(np.sum(weights[list(degrees)]) / total)


def _hpcp(audio: np.ndarray, sample_rate: int) -> _HpcpStats:
    zero = _HpcpStats(
        bins=(0.0,) * 12,
        entropy=0.0,
        std=0.0,
        max=0.0,
        temporal_std=0.0,
    )

    windowing = es.Windowing(type="blackmanharris62", size=_HPCP_FRAME_SIZE)
    spectrum = es.Spectrum(size=_HPCP_FRAME_SIZE)
    peaks = es.SpectralPeaks(
        sampleRate=sample_rate,
        maxPeaks=_HPCP_MAX_PEAKS,
        minFrequency=_HPCP_MIN_FREQUENCY,
        maxFrequency=_HPCP_MAX_FREQUENCY,
        magnitudeThreshold=_HPCP_MAGNITUDE_THRESHOLD,
        orderBy="magnitude",
    )
    hpcp_algo = es.HPCP(
        size=_HPCP_SIZE,
        sampleRate=sample_rate,
        minFrequency=_HPCP_MIN_FREQUENCY,
        maxFrequency=_HPCP_MAX_FREQUENCY,
        normalized="unitSum",
    )

    frames = []
    for start in range(0, len(audio) - _HPCP_FRAME_SIZE + 1, _HPCP_HOP_SIZE):
        spectrum_values = spectrum(windowing(audio[start : start + _HPCP_FRAME_SIZE]))
        frequencies, magnitudes = peaks(spectrum_values)
        frames.append(hpcp_algo(frequencies, magnitudes))

    if len(frames) == 0:
        return zero

    matrix = np.asarray(frames, dtype=np.float32)
    mean = np.mean(matrix, axis=0)
    temporal_std = np.std(matrix, axis=0)

    total = float(np.sum(mean))
    if total > 0:
        probabilities = np.clip(mean / total, _EPS, 1.0)
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    else:
        entropy = 0.0

    return _HpcpStats(
        bins=tuple(float(value) for value in mean),
        entropy=entropy,
        std=float(np.std(mean)),
        max=float(np.max(mean)),
        temporal_std=float(np.mean(temporal_std)),
    )


def _key(audio: np.ndarray, sample_rate: int) -> _KeyStats:
    extractor = es.KeyExtractor(sampleRate=sample_rate)
    key, scale, strength = extractor(audio)

    index = _KEY_INDEX.get(key, 0)
    angle = (index / 12.0) * 2 * np.pi

    return _KeyStats(
        strength=float(strength),
        is_minor=1.0 if scale == "minor" else 0.0,
        cos=float(np.cos(angle)),
        sin=float(np.sin(angle)),
    )
