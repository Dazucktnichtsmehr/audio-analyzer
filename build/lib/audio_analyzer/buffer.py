from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Thread-sicherer Ringpuffer für Live-Audiosamples."""

    def __init__(self, size: int) -> None:
        if size < 4:
            raise ValueError("Ringpuffer muss mindestens 4 Samples groß sein.")

        self._buffer = np.zeros(size, dtype=np.float64)
        self._size = size
        self._write_index = 0
        self._filled = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return self._size

    def write(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            return
        if samples.size >= self._size:
            samples = samples[-self._size :]

        n = samples.size
        with self._lock:
            end = self._write_index + n
            if end <= self._size:
                self._buffer[self._write_index:end] = samples
            else:
                first = self._size - self._write_index
                self._buffer[self._write_index:] = samples[:first]
                self._buffer[: end - self._size] = samples[first:]

            self._write_index = end % self._size
            self._filled = min(self._size, self._filled + n)

    def snapshot(self) -> np.ndarray:
        """Gibt die neuesten Samples chronologisch sortiert zurück."""
        with self._lock:
            if self._filled < self._size:
                out = np.zeros(self._size, dtype=np.float64)
                if self._filled > 0:
                    out[-self._filled :] = self._buffer[: self._filled]
                return out

            if self._write_index == 0:
                return self._buffer.copy()

            return np.concatenate(
                (self._buffer[self._write_index :], self._buffer[: self._write_index])
            )
