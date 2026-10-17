from __future__ import annotations

from functools import lru_cache

import numpy as np

from .constants import (
    DB_FLOOR,
    DEFAULT_STFT_OVERLAP,
    DEFAULT_STFT_SIZE,
    THIRD_OCTAVE_CENTERS_HZ,
)
from .models import AudioData, Spectrogram, Spectrum


@lru_cache(maxsize=16)
def hann_window_stats(size: int) -> tuple[np.ndarray, float, float]:
    """Cached Hann window, coherent sum and ENBW in FFT bins."""
    if size < 2:
        raise ValueError("Hann-Fenstergröße muss mindestens 2 sein.")
    window = np.hanning(size)
    window_sum = float(np.sum(window))
    if window_sum <= 0:
        raise ValueError("Hann-Fenster konnte nicht gebildet werden.")
    enbw_bins = size * float(np.sum(window * window)) / (window_sum * window_sum)
    return window, window_sum, enbw_bins


def calculate_spectrum_from_samples(samples: np.ndarray, sample_rate_hz: int) -> Spectrum:
    """Berechnet ein einseitiges Hann-Fenster-Amplitudenspektrum in dBFS."""
    x = np.asarray(samples, dtype=np.float64).copy()
    n = x.size
    if n < 4:
        raise ValueError("Mindestens 4 Samples werden für eine FFT benötigt.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz muss größer als 0 sein.")

    x -= np.mean(x)
    window, window_sum, _ = hann_window_stats(n)
    fft_values = np.fft.rfft(x * window)
    frequencies_hz = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

    amplitude = np.abs(fft_values) / window_sum
    if n % 2 == 0:
        if amplitude.size > 2:
            amplitude[1:-1] *= 2.0
    elif amplitude.size > 1:
        amplitude[1:] *= 2.0

    floor_linear = 10.0 ** (DB_FLOOR / 20.0)
    dbfs = 20.0 * np.log10(np.maximum(amplitude, floor_linear))

    return Spectrum(
        frequencies_hz=frequencies_hz,
        amplitude=amplitude,
        dbfs=dbfs,
        resolution_hz=sample_rate_hz / n,
        fft_size=n,
    )


def calculate_spectrum(audio: AudioData) -> Spectrum:
    return calculate_spectrum_from_samples(audio.samples, audio.sample_rate_hz)


def calculate_spectrogram_from_samples(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    fft_size: int = DEFAULT_STFT_SIZE,
    hop_size: int | None = None,
    start_s: float = 0.0,
    batch_frames: int = 512,
) -> Spectrogram:
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz muss größer als 0 sein.")
    if fft_size < 256:
        raise ValueError("STFT-FFT-Größe muss mindestens 256 sein.")
    if hop_size is None:
        hop_size = max(1, int(round(fft_size * (1.0 - DEFAULT_STFT_OVERLAP))))
    if hop_size < 1 or hop_size > fft_size:
        raise ValueError("hop_size muss zwischen 1 und fft_size liegen.")
    if batch_frames < 1:
        raise ValueError("batch_frames muss mindestens 1 sein.")
    if x.size < 4:
        raise ValueError("Mindestens 4 Samples werden für ein Spektrogramm benötigt.")

    frame_count = 1 if x.size <= fft_size else int(np.ceil((x.size - fft_size) / hop_size)) + 1
    padded_size = (frame_count - 1) * hop_size + fft_size
    if padded_size > x.size:
        padded = np.zeros(padded_size, dtype=np.float64)
        padded[: x.size] = x
        x = padded

    window, window_sum, _ = hann_window_stats(fft_size)
    frequencies_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate_hz)
    spectrogram_dbfs = np.empty((frequencies_hz.size, frame_count), dtype=np.float32)
    floor_linear = 10.0 ** (DB_FLOOR / 20.0)

    for first_frame in range(0, frame_count, batch_frames):
        last_frame = min(frame_count, first_frame + batch_frames)
        current_count = last_frame - first_frame
        frame_starts = np.arange(first_frame, last_frame, dtype=np.int64) * hop_size
        frames = np.empty((current_count, fft_size), dtype=np.float64)
        for row, frame_start in enumerate(frame_starts):
            frames[row] = x[frame_start : frame_start + fft_size]

        frames -= np.mean(frames, axis=1, keepdims=True)
        frames *= window
        fft_values = np.fft.rfft(frames, axis=1)
        amplitude = np.abs(fft_values) / window_sum
        if fft_size % 2 == 0:
            if amplitude.shape[1] > 2:
                amplitude[:, 1:-1] *= 2.0
        elif amplitude.shape[1] > 1:
            amplitude[:, 1:] *= 2.0

        dbfs = 20.0 * np.log10(np.maximum(amplitude, floor_linear))
        spectrogram_dbfs[:, first_frame:last_frame] = dbfs.T.astype(np.float32, copy=False)

    frame_centers = np.arange(frame_count, dtype=np.float64) * hop_size + fft_size / 2.0
    times_s = start_s + frame_centers / sample_rate_hz
    return Spectrogram(
        times_s=times_s,
        frequencies_hz=frequencies_hz,
        dbfs=spectrogram_dbfs,
        resolution_hz=sample_rate_hz / fft_size,
        fft_size=fft_size,
        hop_size=hop_size,
        time_step_s=hop_size / sample_rate_hz,
    )


def calculate_spectrogram(
    audio: AudioData,
    *,
    fft_size: int = DEFAULT_STFT_SIZE,
    hop_size: int | None = None,
) -> Spectrogram:
    return calculate_spectrogram_from_samples(
        audio.samples,
        audio.sample_rate_hz,
        fft_size=fft_size,
        hop_size=hop_size,
        start_s=audio.start_s,
    )


def strongest_bin(
    spectrum: Spectrum,
    min_frequency_hz: float = 20.0,
    max_frequency_hz: float | None = None,
) -> tuple[float, float, float]:
    mask = spectrum.frequencies_hz >= min_frequency_hz
    if max_frequency_hz is not None:
        mask &= spectrum.frequencies_hz <= max_frequency_hz
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Im gewählten Frequenzbereich liegen keine FFT-Bins.")
    idx = indices[np.argmax(spectrum.amplitude[indices])]
    return (
        float(spectrum.frequencies_hz[idx]),
        float(spectrum.amplitude[idx]),
        float(spectrum.dbfs[idx]),
    )


def build_third_octave_band_bins(
    frequencies_hz: np.ndarray,
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies_hz.ndim != 1 or frequencies_hz.size < 2:
        raise ValueError("Ungültiges FFT-Frequenzraster für die RTA-Auswertung.")

    nyquist_hz = float(frequencies_hz[-1])
    factor = 2.0 ** (1.0 / 6.0)
    centers = THIRD_OCTAVE_CENTERS_HZ.copy()
    lower_edges = centers / factor
    upper_edges = centers * factor
    valid = (
        (centers >= min_frequency_hz)
        & (centers <= max_frequency_hz)
        & (lower_edges > 0.0)
        & (upper_edges <= nyquist_hz)
    )
    centers = centers[valid]
    lower_edges = lower_edges[valid]
    upper_edges = upper_edges[valid]

    bin_indices: list[np.ndarray] = []
    kept_centers: list[float] = []
    kept_lower: list[float] = []
    kept_upper: list[float] = []
    for center, lower, upper in zip(centers, lower_edges, upper_edges):
        indices = np.flatnonzero((frequencies_hz >= lower) & (frequencies_hz < upper))
        if indices.size == 0:
            continue
        bin_indices.append(indices)
        kept_centers.append(float(center))
        kept_lower.append(float(lower))
        kept_upper.append(float(upper))

    if not bin_indices:
        raise ValueError(
            "Mit der gewählten FFT-Größe liegen keine vollständigen "
            "1/3-Oktavbänder im Messbereich."
        )
    return (
        np.asarray(kept_centers, dtype=np.float64),
        np.asarray(kept_lower, dtype=np.float64),
        np.asarray(kept_upper, dtype=np.float64),
        bin_indices,
    )


def calculate_third_octave_power(spectrum: Spectrum, band_bins: list[np.ndarray]) -> np.ndarray:
    bin_power = np.square(spectrum.amplitude, dtype=np.float64)
    _, _, enbw_bins = hann_window_stats(spectrum.fft_size)
    return np.asarray(
        [float(np.sum(bin_power[indices])) / enbw_bins for indices in band_bins],
        dtype=np.float64,
    )


def power_to_db(power: np.ndarray | float, floor_db: float = DB_FLOOR):
    floor_power = 10.0 ** (floor_db / 10.0)
    return 10.0 * np.log10(np.maximum(power, floor_power))
