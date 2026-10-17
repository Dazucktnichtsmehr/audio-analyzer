from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate_hz: int
    start_s: float
    duration_s: float
    source_path: Path
    source_channels: int
    decoder: str


@dataclass(frozen=True)
class Spectrum:
    frequencies_hz: np.ndarray
    amplitude: np.ndarray
    dbfs: np.ndarray
    resolution_hz: float
    fft_size: int


@dataclass(frozen=True)
class Spectrogram:
    times_s: np.ndarray
    frequencies_hz: np.ndarray
    dbfs: np.ndarray
    resolution_hz: float
    fft_size: int
    hop_size: int
    time_step_s: float
