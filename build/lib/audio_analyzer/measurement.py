from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from .buffer import RingBuffer
from .constants import (
    DB_FLOOR,
    DEFAULT_MIN_LOG_FREQUENCY_HZ,
    MEASUREMENT_RTA_FLOOR_DB,
    MEASUREMENT_SPECTRUM_FLOOR_DB,
    RTA_TIME_CONSTANTS_S,
)
from .dsp import (
    build_third_octave_band_bins,
    calculate_spectrum_from_samples,
    calculate_third_octave_power,
    power_to_db,
)
from .plotting import configure_log_audio_axis
from .realtime import import_sounddevice, normalize_device_argument


def _format_rta_frequency(frequency_hz: float) -> str:
    if frequency_hz >= 1000.0:
        value = frequency_hz / 1000.0
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))}k"
        return f"{value:g}k"
    return f"{frequency_hz:g}"


def run_measurement_mode(
    *,
    device: str | None,
    sample_rate_hz: int | None,
    channels: int,
    fft_size: int,
    fps: float,
    smoothing: float,
    max_frequency_hz: float,
    rta_averaging: str,
    delta_range_db: float,
) -> None:
    sd = import_sounddevice()
    if fft_size < 2048:
        raise ValueError(
            "Im Einmessmodus sollte --fft-size mindestens 2048 sein; "
            "empfohlen sind 32768 Samples."
        )
    if fps <= 0 or fps > 120:
        raise ValueError("--fps muss > 0 und <= 120 sein.")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("--smoothing muss im Bereich (0, 1] liegen.")
    if channels < 1:
        raise ValueError("--channels muss mindestens 1 sein.")
    if rta_averaging not in RTA_TIME_CONSTANTS_S:
        raise ValueError("--rta-averaging muss 'fast' oder 'slow' sein.")
    if delta_range_db <= 0:
        raise ValueError("--delta-range muss größer als 0 dB sein.")

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

    sd.check_input_settings(
        device=selected_device,
        channels=channels,
        dtype="float32",
        samplerate=sample_rate_hz,
    )
    display_max_hz = min(max_frequency_hz, sample_rate_hz / 2.0)
    if display_max_hz <= DEFAULT_MIN_LOG_FREQUENCY_HZ:
        raise ValueError("Samplerate ist zu niedrig für den Messbereich.")

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
    spectrum_mask = (
        (initial_spectrum.frequencies_hz >= DEFAULT_MIN_LOG_FREQUENCY_HZ)
        & (initial_spectrum.frequencies_hz <= display_max_hz)
    )
    plot_frequencies = initial_spectrum.frequencies_hz[spectrum_mask]
    if plot_frequencies.size == 0:
        raise ValueError("Keine FFT-Bins im sichtbaren Frequenzbereich.")

    band_centers, band_lower_edges, band_upper_edges, band_bins = build_third_octave_band_bins(
        initial_spectrum.frequencies_hz,
        min_frequency_hz=DEFAULT_MIN_LOG_FREQUENCY_HZ,
        max_frequency_hz=display_max_hz,
    )
    lowest_band_bins = int(band_bins[0].size)

    fig, (ax_spectrum, ax_rta, ax_delta) = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [1.25, 1.0, 0.8]},
    )
    fig.subplots_adjust(bottom=0.13, hspace=0.42)

    spectrum_line, = ax_spectrum.plot(
        plot_frequencies,
        np.full(plot_frequencies.size, MEASUREMENT_SPECTRUM_FLOOR_DB),
        linewidth=1.0,
        label="Live",
    )
    peak_hold_line, = ax_spectrum.plot(
        plot_frequencies,
        np.full(plot_frequencies.size, np.nan),
        linewidth=0.8,
        linestyle="--",
        label="Peak Hold",
    )
    ax_spectrum.set_title("Live-Spektrum")
    ax_spectrum.set_xlabel("Frequenz [Hz]")
    ax_spectrum.set_ylabel("Pegel [dBFS]")
    ax_spectrum.set_ylim(MEASUREMENT_SPECTRUM_FLOOR_DB, 3.0)
    configure_log_audio_axis(
        ax_spectrum, DEFAULT_MIN_LOG_FREQUENCY_HZ, display_max_hz
    )
    ax_spectrum.legend(loc="lower left")
    status_text = ax_spectrum.text(
        0.99, 0.97, "", transform=ax_spectrum.transAxes, ha="right", va="top"
    )

    rta_line, = ax_rta.plot(
        band_centers,
        np.full(band_centers.size, np.nan),
        marker="o",
        linewidth=1.2,
        label="RTA aktuell",
    )
    reference_line, = ax_rta.plot(
        band_centers,
        np.full(band_centers.size, np.nan),
        linewidth=1.0,
        linestyle="--",
        label="Referenz",
    )
    ax_rta.set_title(f"1/3-Oktav-RTA – {rta_averaging.capitalize()} Averaging")
    ax_rta.set_xlabel("Mittenfrequenz [Hz]")
    ax_rta.set_ylabel("Bandpegel [dBFS]")
    ax_rta.set_ylim(MEASUREMENT_RTA_FLOOR_DB, 3.0)
    configure_log_audio_axis(
        ax_rta, float(band_lower_edges[0]), float(band_upper_edges[-1])
    )
    ax_rta.legend(loc="lower left")

    delta_line, = ax_delta.plot(
        band_centers,
        np.full(band_centers.size, np.nan),
        marker="o",
        linewidth=1.0,
    )
    ax_delta.axhline(0.0, linewidth=0.8)
    ax_delta.set_title("Differenz zur Referenz – noch keine Referenz gesetzt")
    ax_delta.set_xlabel("Mittenfrequenz [Hz]")
    ax_delta.set_ylabel("Δ Pegel [dB]")
    ax_delta.set_ylim(-delta_range_db, delta_range_db)
    configure_log_audio_axis(
        ax_delta, float(band_lower_edges[0]), float(band_upper_edges[-1])
    )

    ref_button = Button(fig.add_axes([0.17, 0.025, 0.17, 0.045]), "Referenz setzen")
    clear_ref_button = Button(fig.add_axes([0.37, 0.025, 0.17, 0.045]), "Referenz löschen")
    clear_peak_button = Button(fig.add_axes([0.57, 0.025, 0.20, 0.045]), "Peak Hold löschen")

    tau_s = RTA_TIME_CONSTANTS_S[rta_averaging]
    floor_power = 10.0 ** (DB_FLOOR / 10.0)
    smoothed_spectrum_power: np.ndarray | None = None
    smoothed_band_power: np.ndarray | None = None
    state: dict[str, object] = {
        "current_band_power": None,
        "reference_band_power": None,
        "peak_hold_db": np.full(plot_frequencies.size, -np.inf, dtype=np.float64),
    }

    def set_reference(event) -> None:
        current = state["current_band_power"]
        if current is None:
            print("Referenz kann noch nicht gesetzt werden: warte auf Messdaten.")
            return
        reference = np.asarray(current, dtype=np.float64).copy()
        state["reference_band_power"] = reference
        reference_line.set_ydata(power_to_db(reference))
        ax_delta.set_title("Differenz zur Referenz")
        print("RTA-Referenz wurde auf den aktuellen gemittelten Zustand gesetzt.")
        fig.canvas.draw_idle()

    def clear_reference(event) -> None:
        state["reference_band_power"] = None
        reference_line.set_ydata(np.full(band_centers.size, np.nan))
        delta_line.set_ydata(np.full(band_centers.size, np.nan))
        ax_delta.set_title("Differenz zur Referenz – noch keine Referenz gesetzt")
        print("RTA-Referenz gelöscht.")
        fig.canvas.draw_idle()

    def clear_peak_hold(event) -> None:
        state["peak_hold_db"] = np.full(plot_frequencies.size, -np.inf, dtype=np.float64)
        peak_hold_line.set_ydata(np.full(plot_frequencies.size, np.nan))
        print("Peak Hold gelöscht.")
        fig.canvas.draw_idle()

    ref_button.on_clicked(set_reference)
    clear_ref_button.on_clicked(clear_reference)
    clear_peak_button.on_clicked(clear_peak_hold)

    fig.suptitle(f"PA-Einmessmodus – {device_info['name']} – {sample_rate_hz} Hz")
    print("\n--- PA-Einmessmodus ---")
    print(f"Eingabegerät:          {device_info['name']}")
    print(f"Samplerate:            {sample_rate_hz} Hz")
    print(f"Kanäle:                {channels}")
    print(f"FFT-Größe:             {fft_size:,}")
    print(f"Frequenzauflösung:     {sample_rate_hz / fft_size:.3f} Hz")
    print(f"FFT-Zeitfenster:       {fft_size / sample_rate_hz * 1000.0:.1f} ms")
    print("RTA:                   1/3 Oktave, Z/unbewertet")
    print(f"RTA-Averaging:         {rta_averaging} ({tau_s:.3f} s)")
    print(f"Tiefstes RTA-Band:     {_format_rta_frequency(band_centers[0])} Hz")
    print(f"Bins im tiefsten Band: {lowest_band_bins}")
    print("\nFür Frequenzgangmessungen möglichst Pink Noise verwenden.")
    print(
        "Die Anzeige ist dBFS-basiert und nicht absolut auf dB SPL kalibriert; "
        "Pegeländerungen und Referenzdifferenzen in dB bleiben aber gültig."
    )
    print("Beenden: Diagramm schließen oder Ctrl+C\n")

    last_rta_update = time.monotonic()
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
                spectrum = calculate_spectrum_from_samples(samples, sample_rate_hz)
                amplitude = spectrum.amplitude[spectrum_mask]
                spectrum_power = amplitude * amplitude
                smoothed_spectrum_power = (
                    spectrum_power.copy()
                    if smoothed_spectrum_power is None
                    else smoothing * spectrum_power
                    + (1.0 - smoothing) * smoothed_spectrum_power
                )
                smoothed_spectrum_db = power_to_db(smoothed_spectrum_power)
                spectrum_line.set_ydata(smoothed_spectrum_db)

                current_peak_hold = np.asarray(state["peak_hold_db"], dtype=np.float64)
                current_peak_hold = np.maximum(current_peak_hold, smoothed_spectrum_db)
                state["peak_hold_db"] = current_peak_hold
                peak_hold_line.set_ydata(current_peak_hold)

                raw_band_power = calculate_third_octave_power(spectrum, band_bins)
                rta_now = time.monotonic()
                dt_s = max(1e-6, rta_now - last_rta_update)
                last_rta_update = rta_now
                alpha = 1.0 - np.exp(-dt_s / tau_s)
                smoothed_band_power = (
                    raw_band_power.copy()
                    if smoothed_band_power is None
                    else alpha * raw_band_power + (1.0 - alpha) * smoothed_band_power
                )
                state["current_band_power"] = smoothed_band_power.copy()
                rta_line.set_ydata(power_to_db(smoothed_band_power))

                reference = state["reference_band_power"]
                if reference is not None:
                    reference_band_power = np.asarray(reference, dtype=np.float64)
                    delta_db = 10.0 * np.log10(
                        np.maximum(smoothed_band_power, floor_power)
                        / np.maximum(reference_band_power, floor_power)
                    )
                    delta_line.set_ydata(delta_db)

                peak_sample = float(np.max(np.abs(samples)))
                peak_index = int(np.argmax(smoothed_spectrum_power))
                peak_frequency = float(plot_frequencies[peak_index])
                peak_db = float(smoothed_spectrum_db[peak_index])
                input_status = (
                    "CLIP!"
                    if peak_sample >= 0.999
                    else "Input hoch"
                    if peak_sample >= 0.9
                    else "Input sehr niedrig"
                    if peak_sample < 0.005
                    else "Input OK"
                )
                status_text.set_text(
                    f"Peak: {peak_frequency:.1f} Hz / {peak_db:.1f} dBFS\n"
                    f"Sample Peak: {20.0 * np.log10(max(peak_sample, 1e-12)):.1f} dBFS\n"
                    f"{input_status}"
                )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("\nEinmessmodus beendet.")
