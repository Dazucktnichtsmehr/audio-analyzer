from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from .buffer import RingBuffer
from .constants import DB_FLOOR, DEFAULT_MIN_LOG_FREQUENCY_HZ
from .dsp import calculate_spectrum_from_samples
from .plotting import configure_log_audio_axis


def import_sounddevice():
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Für Live- und Einmessmodus fehlt 'sounddevice'. Installieren mit: "
            "pip install sounddevice"
        ) from exc
    return sd


def normalize_device_argument(device: str | None):
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        return device


def list_input_devices() -> None:
    sd = import_sounddevice()
    print("\n--- Audio-Eingabegeräte ---")
    found = False
    for index, device in enumerate(sd.query_devices()):
        max_inputs = int(device["max_input_channels"])
        if max_inputs <= 0:
            continue
        found = True
        print(
            f"{index:>3}: {device['name']} | Inputs: {max_inputs} "
            f"| Default SR: {float(device['default_samplerate']):.0f} Hz"
        )
    if not found:
        print("Keine Audio-Eingabegeräte gefunden.")


def run_live_mode(
    *,
    device: str | None,
    sample_rate_hz: int | None,
    channels: int,
    fft_size: int,
    fps: float,
    smoothing: float,
    max_frequency_hz: float,
    waveform_ms: float,
) -> None:
    sd = import_sounddevice()
    if fft_size < 256:
        raise ValueError("--fft-size muss mindestens 256 sein.")
    if fps <= 0 or fps > 120:
        raise ValueError("--fps muss > 0 und <= 120 sein.")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("--smoothing muss im Bereich (0, 1] liegen.")
    if channels < 1:
        raise ValueError("--channels muss mindestens 1 sein.")
    if waveform_ms <= 0:
        raise ValueError("--waveform-ms muss größer als 0 sein.")

    selected_device = normalize_device_argument(device)
    device_info = sd.query_devices(selected_device, "input")
    max_input_channels = int(device_info["max_input_channels"])
    if max_input_channels < channels:
        raise ValueError(
            f"Das Eingabegerät unterstützt maximal {max_input_channels} "
            f"Eingangskanäle, angefordert wurden {channels}."
        )
    if sample_rate_hz is None:
        sample_rate_hz = int(round(float(device_info["default_samplerate"])))
    if sample_rate_hz <= 0:
        raise ValueError("Samplerate muss größer als 0 sein.")

    nyquist_hz = sample_rate_hz / 2.0
    display_max_hz = min(max_frequency_hz, nyquist_hz)
    if display_max_hz <= DEFAULT_MIN_LOG_FREQUENCY_HZ:
        raise ValueError("Samplerate ist zu niedrig für die Audioanzeige.")

    sd.check_input_settings(
        device=selected_device,
        channels=channels,
        dtype="float32",
        samplerate=sample_rate_hz,
    )

    ring = RingBuffer(fft_size)
    callback_status_last_print = 0.0

    def audio_callback(indata, frames, time_info, status) -> None:
        nonlocal callback_status_last_print
        if status:
            now = time.monotonic()
            if now - callback_status_last_print > 1.0:
                print(f"Audio-Status: {status}")
                callback_status_last_print = now
        mono = (
            np.mean(indata, axis=1, dtype=np.float64)
            if indata.ndim == 2
            else np.asarray(indata, dtype=np.float64)
        )
        ring.write(mono)

    initial_spectrum = calculate_spectrum_from_samples(
        np.zeros(fft_size, dtype=np.float64), sample_rate_hz
    )
    frequency_mask = (
        (initial_spectrum.frequencies_hz >= DEFAULT_MIN_LOG_FREQUENCY_HZ)
        & (initial_spectrum.frequencies_hz <= display_max_hz)
    )
    plot_frequencies = initial_spectrum.frequencies_hz[frequency_mask]
    if plot_frequencies.size == 0:
        raise ValueError("Keine FFT-Bins im gewählten Darstellungsbereich.")

    waveform_samples = min(
        fft_size,
        max(32, int(round(sample_rate_hz * waveform_ms / 1000.0))),
    )
    waveform_time_ms = (
        np.arange(-waveform_samples + 1, 1, dtype=np.float64)
        / sample_rate_hz
        * 1000.0
    )

    fig, (ax_wave, ax_spectrum) = plt.subplots(2, 1, figsize=(13, 8))
    wave_line, = ax_wave.plot(waveform_time_ms, np.zeros(waveform_samples), linewidth=0.9)
    ax_wave.set_title("Live-Zeitsignal")
    ax_wave.set_xlabel("Zeit relativ zu jetzt [ms]")
    ax_wave.set_ylabel("Amplitude [FS]")
    ax_wave.set_xlim(waveform_time_ms[0], 0)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.grid(True, alpha=0.3)

    spectrum_line, = ax_spectrum.plot(
        plot_frequencies,
        np.full(plot_frequencies.size, DB_FLOOR),
        linewidth=1.0,
    )
    ax_spectrum.set_title("Live-Spektrum")
    ax_spectrum.set_xlabel("Frequenz [Hz]")
    ax_spectrum.set_ylabel("Pegel [dBFS]")
    ax_spectrum.set_ylim(DB_FLOOR, 3.0)
    configure_log_audio_axis(
        ax_spectrum,
        DEFAULT_MIN_LOG_FREQUENCY_HZ,
        display_max_hz,
    )
    info_text = ax_spectrum.text(
        0.99, 0.97, "", transform=ax_spectrum.transAxes, ha="right", va="top"
    )
    fig.suptitle(f"Live Audio Analyzer – {device_info['name']} – {sample_rate_hz} Hz")
    fig.tight_layout()

    smoothed_power: np.ndarray | None = None
    floor_power = (10.0 ** (DB_FLOOR / 20.0)) ** 2

    print("\n--- Live-Modus ---")
    print(f"Eingabegerät:          {device_info['name']}")
    print(f"Samplerate:            {sample_rate_hz} Hz")
    print(f"Kanäle:                {channels}")
    print(f"FFT-Größe:             {fft_size}")
    print(f"Frequenzauflösung:     {sample_rate_hz / fft_size:.3f} Hz")
    print(f"FFT-Zeitfenster:       {fft_size / sample_rate_hz * 1000.0:.1f} ms")
    print(f"Anzeige-Updates:       {fps:.1f} FPS")
    print(f"Glättung:              {smoothing:.2f}")
    print("Beenden: Diagramm schließen oder Ctrl+C\n")

    try:
        with sd.InputStream(
            device=selected_device,
            channels=channels,
            samplerate=sample_rate_hz,
            dtype="float32",
            blocksize=0,
            callback=audio_callback,
        ):
            plt.show(block=False)
            update_interval_s = 1.0 / fps
            next_update = time.monotonic()
            while plt.fignum_exists(fig.number):
                now = time.monotonic()
                if now < next_update:
                    plt.pause(min(next_update - now, 0.01))
                    continue
                next_update = now + update_interval_s
                samples = ring.snapshot()
                wave_line.set_ydata(samples[-waveform_samples:])
                spectrum = calculate_spectrum_from_samples(samples, sample_rate_hz)
                amplitude = spectrum.amplitude[frequency_mask]
                power = amplitude * amplitude
                smoothed_power = (
                    power
                    if smoothed_power is None
                    else smoothing * power + (1.0 - smoothing) * smoothed_power
                )
                smoothed_db = 10.0 * np.log10(np.maximum(smoothed_power, floor_power))
                spectrum_line.set_ydata(smoothed_db)
                peak_index = int(np.argmax(smoothed_power))
                info_text.set_text(
                    f"Peak: {float(plot_frequencies[peak_index]):.1f} Hz\n"
                    f"{float(smoothed_db[peak_index]):.1f} dBFS"
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("\nLive-Modus beendet.")
