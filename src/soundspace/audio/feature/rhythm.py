from dataclasses import dataclass

import essentia.standard as es
import librosa
import numpy as np

from .io import SAMPLE_RATE

_EPS = 1e-12
_RHYTHM_FRAME_SIZE = 2048
_RHYTHM_HOP_SIZE = 1024
_MEL_BANDS = 40
_RHYTHM_TRANSFORM_FRAME_SIZE = 8
_RHYTHM_TRANSFORM_HOP_SIZE = 4
_MIN_RHYTHM_FRAMES = 8
_BEAT_LOUDNESS_BANDS = (20, 150, 400, 3200, 7000)


@dataclass(frozen=True, slots=True)
class RhythmFeatures:
    tempo_bpm: float
    onset_strength_mean: float
    onset_strength_std: float
    beat_interval_mean: float
    beat_interval_std: float
    beat_interval_cv: float
    bpm_first_peak: float
    bpm_first_weight: float
    bpm_first_spread: float
    bpm_second_peak: float
    bpm_second_weight: float
    bpm_second_spread: float
    bpm_peak_ratio: float
    bpm_histogram_entropy: float
    rhythm_transform_mean: float
    rhythm_transform_std: float
    rhythm_transform_max: float
    rhythm_transform_entropy: float
    beats_loudness_mean: float
    beats_loudness_std: float
    beats_low_ratio: float
    beats_mid_ratio: float
    beats_high_ratio: float


@dataclass(frozen=True, slots=True)
class _IntervalStats:
    mean: float
    std: float
    cv: float


@dataclass(frozen=True, slots=True)
class _BpmHistogram:
    first_peak: float
    first_weight: float
    first_spread: float
    second_peak: float
    second_weight: float
    second_spread: float
    peak_ratio: float
    entropy: float


@dataclass(frozen=True, slots=True)
class _RhythmTransform:
    mean: float
    std: float
    max: float
    entropy: float


@dataclass(frozen=True, slots=True)
class _BeatsLoudness:
    mean: float
    std: float
    low_ratio: float
    mid_ratio: float
    high_ratio: float


def compute_rhythm(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> RhythmFeatures:
    audio = np.asarray(audio, dtype=np.float32)

    extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, _, _, beats_intervals = extractor(audio)

    onset_mean, onset_std = _onset_stats(audio, sample_rate)
    intervals = _interval_stats(beats_intervals)
    histogram = _bpm_histogram(beats_intervals)
    transform = _rhythm_transform(audio, sample_rate)
    loudness = _beats_loudness(audio, sample_rate, beats)

    return RhythmFeatures(
        tempo_bpm=float(bpm),
        onset_strength_mean=onset_mean,
        onset_strength_std=onset_std,
        beat_interval_mean=intervals.mean,
        beat_interval_std=intervals.std,
        beat_interval_cv=intervals.cv,
        bpm_first_peak=histogram.first_peak,
        bpm_first_weight=histogram.first_weight,
        bpm_first_spread=histogram.first_spread,
        bpm_second_peak=histogram.second_peak,
        bpm_second_weight=histogram.second_weight,
        bpm_second_spread=histogram.second_spread,
        bpm_peak_ratio=histogram.peak_ratio,
        bpm_histogram_entropy=histogram.entropy,
        rhythm_transform_mean=transform.mean,
        rhythm_transform_std=transform.std,
        rhythm_transform_max=transform.max,
        rhythm_transform_entropy=transform.entropy,
        beats_loudness_mean=loudness.mean,
        beats_loudness_std=loudness.std,
        beats_low_ratio=loudness.low_ratio,
        beats_mid_ratio=loudness.mid_ratio,
        beats_high_ratio=loudness.high_ratio,
    )


def _onset_stats(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    values = np.asarray(onset, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _interval_stats(beats_intervals: np.ndarray) -> _IntervalStats:
    intervals = np.asarray(beats_intervals, dtype=np.float32)
    if intervals.size < 2:
        return _IntervalStats(mean=0.0, std=0.0, cv=0.0)
    mean = float(np.mean(intervals))
    std = float(np.std(intervals))
    cv = std / mean if mean > 0 else 0.0
    return _IntervalStats(mean=mean, std=std, cv=cv)


def _bpm_histogram(beats_intervals: np.ndarray) -> _BpmHistogram:
    zero = _BpmHistogram(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    intervals = np.asarray(beats_intervals, dtype=np.float32)
    if intervals.size < 2:
        return zero

    descriptors = es.BpmHistogramDescriptors()
    (
        first_peak,
        first_weight,
        first_spread,
        second_peak,
        second_weight,
        second_spread,
        histogram,
    ) = descriptors(intervals)

    peak_ratio = second_weight / first_weight if first_weight > 0 else 0.0

    counts = np.asarray(histogram, dtype=np.float32)
    total = float(np.sum(counts))
    if total > 0:
        probabilities = np.clip(counts / total, _EPS, 1.0)
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    else:
        entropy = 0.0

    return _BpmHistogram(
        first_peak=float(first_peak),
        first_weight=float(first_weight),
        first_spread=float(first_spread),
        second_peak=float(second_peak),
        second_weight=float(second_weight),
        second_spread=float(second_spread),
        peak_ratio=peak_ratio,
        entropy=entropy,
    )


def _rhythm_transform(audio: np.ndarray, sample_rate: int) -> _RhythmTransform:
    zero = _RhythmTransform(0.0, 0.0, 0.0, 0.0)

    windowing = es.Windowing(type="hann", size=_RHYTHM_FRAME_SIZE)
    spectrum = es.Spectrum(size=_RHYTHM_FRAME_SIZE)
    mel_bands = es.MelBands(
        sampleRate=sample_rate,
        numberBands=_MEL_BANDS,
        lowFrequencyBound=0,
        highFrequencyBound=sample_rate / 2,
    )

    mel_frames = []
    for start in range(0, len(audio) - _RHYTHM_FRAME_SIZE + 1, _RHYTHM_HOP_SIZE):
        frame = audio[start : start + _RHYTHM_FRAME_SIZE]
        mel_frames.append(mel_bands(spectrum(windowing(frame))))

    if len(mel_frames) < _MIN_RHYTHM_FRAMES:
        return zero

    mel_matrix = np.asarray(mel_frames, dtype=np.float32)
    transform = es.RhythmTransform(
        frameSize=_RHYTHM_TRANSFORM_FRAME_SIZE,
        hopSize=_RHYTHM_TRANSFORM_HOP_SIZE,
    )
    rhythm_matrix = transform(mel_matrix)

    if rhythm_matrix.size == 0:
        return zero

    values = rhythm_matrix.flatten()
    total = float(np.sum(np.abs(values)))
    if total > 0:
        probabilities = np.clip(np.abs(values) / total, _EPS, 1.0)
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    else:
        entropy = 0.0

    return _RhythmTransform(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        max=float(np.max(values)),
        entropy=entropy,
    )


def _beats_loudness(
    audio: np.ndarray,
    sample_rate: int,
    beats: np.ndarray,
) -> _BeatsLoudness:
    zero = _BeatsLoudness(0.0, 0.0, 0.0, 0.0, 0.0)
    if len(beats) < 2:
        return zero

    loudness_algo = es.BeatsLoudness(
        sampleRate=sample_rate,
        beats=beats,
        frequencyBands=[*_BEAT_LOUDNESS_BANDS, sample_rate / 2],
    )
    loudness, band_ratio = loudness_algo(audio)

    if len(loudness) == 0:
        return zero

    values = np.asarray(loudness, dtype=np.float32)
    bands = np.asarray(band_ratio, dtype=np.float32)

    if bands.ndim == 2 and bands.shape[1] >= 5:
        low = float(np.mean(bands[:, 0] + bands[:, 1]))
        mid = float(np.mean(bands[:, 2] + bands[:, 3]))
        high = float(np.mean(bands[:, 4]))
    else:
        low = mid = high = 0.0

    return _BeatsLoudness(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        low_ratio=low,
        mid_ratio=mid,
        high_ratio=high,
    )
