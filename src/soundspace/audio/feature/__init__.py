from enum import StrEnum

from .io import SAMPLE_RATE, load_audio
from .rhythm import RhythmFeatures, compute_rhythm
from .spectral import SpectralFeatures, compute_spectral
from .tonal import TonalFeatures, compute_tonal


class FeatureGroup(StrEnum):
    rhythm = "rhythm"
    spectral = "spectral"
    tonal = "tonal"


ALL_FEATURE_GROUPS: tuple[FeatureGroup, ...] = tuple(FeatureGroup)


__all__ = [
    "SAMPLE_RATE",
    "load_audio",
    "FeatureGroup",
    "ALL_FEATURE_GROUPS",
    "RhythmFeatures",
    "SpectralFeatures",
    "TonalFeatures",
    "compute_rhythm",
    "compute_spectral",
    "compute_tonal",
]
