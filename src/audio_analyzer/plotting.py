from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    DB_FLOOR,
    DEFAULT_MIN_LOG_FREQUENCY_HZ,
    DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB,
    MAX_WAVEFORM_POINTS,
)
from .models import AudioData, Spectrogram, Spectrum


def downsample_for_plot(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = MAX_WAVEFORM_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    if y.size <= max_points:
        return x, y
    step = int(np.ceil(y.size / max_points))
    return x[::step], y[::step]


def configure_log_audio_axis(axis, min_frequency_hz: float, max_frequency_hz: float) -> None:
    axis.set_xscale("log")
    axis.set_xlim(min_frequency_hz, max_frequency_hz)
    audio_ticks = np.array(
        [20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000],
        dtype=float,
    )
    visible_ticks = audio_ticks[
        (audio_ticks >= min_frequency_hz) & (audio_ticks <= max_frequency_hz)
    ]
    axis.set_xticks(visible_ticks)
    axis.set_xticklabels([
        f"{int(freq / 1000)}k" if freq >= 1000 else f"{int(freq)}"
        for freq in visible_ticks
    ])
    axis.grid(True, which="both", alpha=0.3)


def configure_log_frequency_y_axis(axis, min_frequency_hz: float, max_frequency_hz: float) -> None:
    axis.set_yscale("log")
    axis.set_ylim(min_frequency_hz, max_frequency_hz)
    audio_ticks = np.array(
        [20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000],
        dtype=float,
    )
    visible_ticks = audio_ticks[
        (audio_ticks >= min_frequency_hz) & (audio_ticks <= max_frequency_hz)
    ]
    axis.set_yticks(visible_ticks)
    axis.set_yticklabels([
        f"{int(freq / 1000)}k" if freq >= 1000 else f"{int(freq)}"
        for freq in visible_ticks
    ])


def plot_results(
    audio: AudioData,
    spectrum: Spectrum,
    max_frequency_hz: float,
    min_log_frequency_hz: float = DEFAULT_MIN_LOG_FREQUENCY_HZ,
) -> None:
    nyquist_hz = audio.sample_rate_hz / 2.0
    display_max_hz = min(max_frequency_hz, nyquist_hz)
    if display_max_hz <= min_log_frequency_hz:
        raise ValueError("Darstellungsbereich ist zu klein für die logarithmische Achse.")

    time_s = np.arange(audio.samples.size, dtype=np.float64) / audio.sample_rate_hz + audio.start_s
    plot_time_s, plot_samples = downsample_for_plot(time_s, audio.samples)
    frequency_mask = spectrum.frequencies_hz <= display_max_hz
    plot_freq = spectrum.frequencies_hz[frequency_mask]
    plot_amp = spectrum.amplitude[frequency_mask]
    plot_db = spectrum.dbfs[frequency_mask]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    axes[0].plot(plot_time_s, plot_samples, linewidth=0.8)
    axes[0].set_title("Zeitsignal")
    axes[0].set_xlabel("Zeit [s]")
    axes[0].set_ylabel("Amplitude [FS]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(plot_freq, plot_amp, linewidth=0.8)
    axes[1].set_title("Einseitiges Amplitudenspektrum")
    axes[1].set_xlabel("Frequenz [Hz]")
    axes[1].set_ylabel("Amplitude [FS]")
    axes[1].set_xlim(0, display_max_hz)
    axes[1].grid(True, alpha=0.3)

    log_mask = plot_freq >= min_log_frequency_hz
    log_freq = plot_freq[log_mask]
    log_db = plot_db[log_mask]
    axes[2].plot(log_freq, log_db, linewidth=0.8)
    axes[2].set_title("Spektrum in dBFS – logarithmische Frequenzachse")
    axes[2].set_xlabel("Frequenz [Hz]")
    axes[2].set_ylabel("Pegel [dBFS]")
    axes[2].set_ylim(DB_FLOOR, max(3.0, float(np.max(log_db)) + 3.0))
    configure_log_audio_axis(axes[2], min_log_frequency_hz, display_max_hz)

    fig.suptitle(audio.source_path.name)
    fig.tight_layout()
    plt.show()


def plot_spectrogram(
    audio: AudioData,
    spectrogram: Spectrogram,
    *,
    max_frequency_hz: float,
    min_frequency_hz: float = DEFAULT_MIN_LOG_FREQUENCY_HZ,
    dynamic_range_db: float = DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB,
) -> None:
    if dynamic_range_db <= 0:
        raise ValueError("Spektrogramm-Dynamikbereich muss größer als 0 dB sein.")

    nyquist_hz = audio.sample_rate_hz / 2.0
    display_max_hz = min(max_frequency_hz, nyquist_hz)
    if display_max_hz <= min_frequency_hz:
        raise ValueError("Darstellungsbereich ist zu klein für das Spektrogramm.")

    frequency_mask = (
        (spectrogram.frequencies_hz >= min_frequency_hz)
        & (spectrogram.frequencies_hz <= display_max_hz)
    )
    frequencies_hz = spectrogram.frequencies_hz[frequency_mask]
    display_dbfs = spectrogram.dbfs[frequency_mask, :]
    if frequencies_hz.size == 0 or display_dbfs.size == 0:
        raise ValueError("Keine STFT-Bins im gewählten Darstellungsbereich.")

    peak_db = float(np.max(display_dbfs))
    color_max_db = min(0.0, peak_db)
    color_min_db = max(DB_FLOOR, color_max_db - dynamic_range_db)

    fig, axis = plt.subplots(figsize=(14, 7))
    mesh = axis.pcolormesh(
        spectrogram.times_s,
        frequencies_hz,
        display_dbfs,
        shading="auto",
        vmin=color_min_db,
        vmax=color_max_db,
    )
    axis.set_title("Spektrogramm – STFT")
    axis.set_xlabel("Zeit [s]")
    axis.set_ylabel("Frequenz [Hz]")
    configure_log_frequency_y_axis(axis, min_frequency_hz, display_max_hz)
    colorbar = fig.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label("Pegel [dBFS]")
    fig.suptitle(audio.source_path.name)
    fig.tight_layout()
    plt.show()
