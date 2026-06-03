from dataclasses import dataclass

import essentia.standard as es
import librosa
import numpy as np

from .io import SAMPLE_RATE

_N_MFCC = 5
_INHARMONICITY_FRAME_SIZE = 2048
_INHARMONICITY_HOP_SIZE = 512
_MAX_SPECTRAL_PEAKS = 50
_MIN_FREQUENCY = 20


@dataclass(frozen=True, slots=True)
class SpectralFeatures:
    mfcc_1_mean: float
    mfcc_2_mean: float
    mfcc_3_mean: float
    mfcc_4_mean: float
    mfcc_5_mean: float
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_contrast_mean: float
    inharmonicity_mean: float
    inharmonicity_std: float


def compute_spectral(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> SpectralFeatures:
    audio = np.asarray(audio, dtype=np.float32)

    mfcc = _mfcc_means(audio, sample_rate, _N_MFCC)
    centroid_mean, centroid_std = _centroid_stats(audio, sample_rate)
    contrast_mean = _contrast_mean(audio, sample_rate)
    inharmonicity_mean, inharmonicity_std = _inharmonicity_stats(audio, sample_rate)

    return SpectralFeatures(
        mfcc_1_mean=mfcc[0],
        mfcc_2_mean=mfcc[1],
        mfcc_3_mean=mfcc[2],
        mfcc_4_mean=mfcc[3],
        mfcc_5_mean=mfcc[4],
        spectral_centroid_mean=centroid_mean,
        spectral_centroid_std=centroid_std,
        spectral_contrast_mean=contrast_mean,
        inharmonicity_mean=inharmonicity_mean,
        inharmonicity_std=inharmonicity_std,
    )


def _mfcc_means(audio: np.ndarray, sample_rate: int, n_mfcc: int) -> tuple[float, ...]:
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    means = np.asarray(np.mean(mfcc, axis=1), dtype=np.float32).reshape(-1)
    if means.size != n_mfcc:
        raise ValueError(f"unexpected MFCC shape: {mfcc.shape}")
    return tuple(float(value) for value in means)


def _centroid_stats(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    values = np.asarray(centroid, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _contrast_mean(audio: np.ndarray, sample_rate: int) -> float:
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    values = np.asarray(contrast, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _inharmonicity_stats(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    windowing = es.Windowing(type="hann", size=_INHARMONICITY_FRAME_SIZE)
    spectrum = es.Spectrum(size=_INHARMONICITY_FRAME_SIZE)
    peaks = es.SpectralPeaks(
        sampleRate=sample_rate,
        maxPeaks=_MAX_SPECTRAL_PEAKS,
        minFrequency=_MIN_FREQUENCY,
        maxFrequency=sample_rate / 2,
    )
    pitch = es.PitchYinFFT(sampleRate=sample_rate)
    inharmonicity = es.Inharmonicity()

    values = []
    for start in range(
        0, len(audio) - _INHARMONICITY_FRAME_SIZE + 1, _INHARMONICITY_HOP_SIZE
    ):
        spectrum_values = spectrum(
            windowing(audio[start : start + _INHARMONICITY_FRAME_SIZE])
        )
        frequencies, magnitudes = peaks(spectrum_values)
        pitch_hz, _ = pitch(spectrum_values)
        if pitch_hz > 0 and len(frequencies) > 1:
            value = inharmonicity(frequencies, magnitudes)
            if not np.isnan(value) and value >= 0:
                values.append(value)

    if len(values) == 0:
        return 0.0, 0.0

    array = np.asarray(values, dtype=np.float32)
    return float(np.mean(array)), float(np.std(array))
