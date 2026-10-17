from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .constants import (
    DEFAULT_DELTA_RANGE_DB,
    DEFAULT_LIVE_FPS,
    DEFAULT_LIVE_FFT_SIZE,
    DEFAULT_LIVE_SMOOTHING,
    DEFAULT_LIVE_WAVEFORM_MS,
    DEFAULT_MAX_FREQUENCY_HZ,
    DEFAULT_MEASUREMENT_FFT_SIZE,
    DEFAULT_RTA_AVERAGING,
    DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB,
    DEFAULT_STFT_SIZE,
)
from .dsp import calculate_spectrogram, calculate_spectrum, strongest_bin
from .fileio import load_audio_mono
from .measurement import run_measurement_mode
from .plotting import plot_results, plot_spectrogram
from .realtime import list_input_devices, run_live_mode
from .system_audio import list_system_audio_devices, run_system_audio_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-analyzer",
        description=(
            "Audio-Analyzer: FFT, Spektrogramm, Live-Eingänge, PA-Einmessung "
            "und natives Systemaudio auf macOS, Windows und Linux."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("file", nargs="?", type=Path, help="Audiodatei für den Datei-Modus")
    parser.add_argument("--live", action="store_true", help="Live-Analyse eines Audio-Eingangs")
    parser.add_argument(
        "--measure",
        action="store_true",
        help="PA-Einmessmodus: Live-Spektrum + 1/3-Oktav-RTA + Referenzvergleich",
    )
    parser.add_argument(
        "--system-audio",
        action="store_true",
        help="Systemton nativ/ohne virtuelles Audiokabel analysieren",
    )
    parser.add_argument("--list-devices", action="store_true", help="Audio-Eingabegeräte anzeigen")
    parser.add_argument(
        "--list-system-devices",
        action="store_true",
        help="Windows/Linux: wählbare Systemaudio-Ausgänge anzeigen",
    )
    parser.add_argument(
        "--system-device",
        type=str,
        default=None,
        help="Windows/Linux: Ausgangsgerät für --system-audio (Name/ID)",
    )
    parser.add_argument("--device", type=str, default=None, help="Geräteindex oder Name")
    parser.add_argument("--samplerate", type=int, default=None, help="Samplerate in Hz")
    parser.add_argument("--channels", type=int, default=None, help="Anzahl Eingangskanäle")
    parser.add_argument(
        "--fft-size",
        type=int,
        default=None,
        help="FFT-Größe; Standard 8192 live/system, 32768 measure",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_LIVE_FPS, help="Anzeige-Updates/s")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=DEFAULT_LIVE_SMOOTHING,
        help="Live-Glättung: 1 = ungeglättet, kleiner = ruhiger",
    )
    parser.add_argument(
        "--waveform-ms",
        type=float,
        default=DEFAULT_LIVE_WAVEFORM_MS,
        help="Sichtbare Live-Wellenform in ms",
    )
    parser.add_argument(
        "--rta-averaging",
        choices=("fast", "slow"),
        default=DEFAULT_RTA_AVERAGING,
        help="Nur --measure: fast (~125 ms) oder slow (~1 s)",
    )
    parser.add_argument(
        "--delta-range",
        type=float,
        default=DEFAULT_DELTA_RANGE_DB,
        help="Nur --measure: +/- Referenz-Anzeigebereich in dB",
    )
    parser.add_argument("--spectrogram", action="store_true", help="STFT-Spektrogramm anzeigen")
    parser.add_argument("--stft-size", type=int, default=DEFAULT_STFT_SIZE, help="STFT-FFT-Größe")
    parser.add_argument("--hop-size", type=int, default=None, help="STFT-Schrittweite in Samples")
    parser.add_argument(
        "--spectrogram-range",
        type=float,
        default=DEFAULT_SPECTROGRAM_DYNAMIC_RANGE_DB,
        help="Dynamikbereich des Spektrogramms in dB",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Datei-Startzeit in Sekunden")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Datei-Dauer in Sekunden; Standard 5 s, bei Spektrogramm Restdatei",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=DEFAULT_MAX_FREQUENCY_HZ,
        help="Maximal dargestellte Frequenz in Hz",
    )
    parser.add_argument(
        "--decoder",
        choices=("auto", "soundfile", "ffmpeg"),
        default="auto",
        help="Datei-Decoder",
    )
    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def ask_for_file() -> Path:
    raw = input("Pfad zur Audiodatei: ").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("Kein Dateipfad angegeben.")
    return Path(raw).expanduser()


def run_file_mode(args: argparse.Namespace) -> None:
    path = args.file if args.file is not None else ask_for_file()
    duration_s = (None if args.spectrogram else 5.0) if args.duration is None else args.duration
    audio = load_audio_mono(path, args.start, duration_s, args.decoder)

    if args.spectrogram:
        spectrogram = calculate_spectrogram(audio, fft_size=args.stft_size, hop_size=args.hop_size)
        print("\n--- Spektrogramm-Analyse ---")
        print(f"Datei:                  {audio.source_path}")
        print(f"Decoder:                 {audio.decoder}")
        print(f"Quellkanäle:             {audio.source_channels}")
        print(f"Samplerate:              {audio.sample_rate_hz} Hz")
        print(f"Analysierter Start:      {audio.start_s:.3f} s")
        print(f"Analysierte Dauer:       {audio.duration_s:.3f} s")
        print(f"STFT-FFT-Größe:          {spectrogram.fft_size:,}")
        print(f"Hop-Größe:               {spectrogram.hop_size:,} Samples")
        print(f"Fensterdauer:            {spectrogram.fft_size / audio.sample_rate_hz * 1000.0:.1f} ms")
        print(f"Zeitabstand:             {spectrogram.time_step_s * 1000.0:.1f} ms")
        print(f"Frequenzauflösung:       {spectrogram.resolution_hz:.3f} Hz")
        print(f"Zeitfenster / Frames:    {spectrogram.times_s.size:,}")
        print(f"Nyquist-Frequenz:        {audio.sample_rate_hz / 2.0:.1f} Hz")
        plot_spectrogram(
            audio,
            spectrogram,
            max_frequency_hz=args.max_frequency,
            dynamic_range_db=args.spectrogram_range,
        )
        return

    spectrum = calculate_spectrum(audio)
    peak_frequency_hz, peak_amplitude, peak_dbfs = strongest_bin(
        spectrum,
        min_frequency_hz=20.0,
        max_frequency_hz=min(args.max_frequency, audio.sample_rate_hz / 2.0),
    )
    print("\n--- Analyse ---")
    print(f"Datei:                  {audio.source_path}")
    print(f"Decoder:                 {audio.decoder}")
    print(f"Quellkanäle:             {audio.source_channels}")
    print(f"Samplerate:              {audio.sample_rate_hz} Hz")
    print(f"Analysierter Start:      {audio.start_s:.3f} s")
    print(f"Analysierte Dauer:       {audio.duration_s:.3f} s")
    print(f"Samples / FFT-Größe:     {spectrum.fft_size:,}")
    print(f"Frequenzauflösung:       {spectrum.resolution_hz:.6f} Hz")
    print(f"Nyquist-Frequenz:        {audio.sample_rate_hz / 2.0:.1f} Hz")
    print(f"Stärkster FFT-Bin:       {peak_frequency_hz:.3f} Hz")
    print(f"Amplitude dieses Bins:   {peak_amplitude:.6f} FS")
    print(f"Pegel dieses Bins:       {peak_dbfs:.2f} dBFS")
    plot_results(audio, spectrum, args.max_frequency)


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_frequency <= 0:
        raise ValueError("--max-frequency muss größer als 0 sein.")
    if args.spectrogram_range <= 0:
        raise ValueError("--spectrogram-range muss größer als 0 sein.")
    if args.stft_size < 256:
        raise ValueError("--stft-size muss mindestens 256 sein.")
    if args.hop_size is not None and not 1 <= args.hop_size <= args.stft_size:
        raise ValueError("--hop-size muss zwischen 1 und --stft-size liegen.")
    if args.channels is not None and args.channels < 1:
        raise ValueError("--channels muss mindestens 1 sein.")
    if args.fft_size is not None and args.fft_size < 256:
        raise ValueError("--fft-size muss mindestens 256 sein.")
    if args.delta_range <= 0:
        raise ValueError("--delta-range muss größer als 0 dB sein.")

    realtime_modes = int(args.live) + int(args.measure) + int(args.system_audio)
    if realtime_modes > 1:
        raise ValueError("--live, --measure und --system-audio sind alternative Modi.")
    if args.spectrogram and realtime_modes:
        raise ValueError("--spectrogram kann nicht mit einem Echtzeitmodus kombiniert werden.")
    if args.system_audio and args.device is not None:
        raise ValueError(
            "Für --system-audio bitte --system-device statt --device verwenden."
        )
    if args.system_audio and args.channels is not None:
        raise ValueError(
            "--channels wird bei --system-audio automatisch aus dem Ausgang bestimmt."
        )
    if args.system_device is not None and not args.system_audio:
        raise ValueError("--system-device gehört zu --system-audio.")


def _dispatch(args: argparse.Namespace) -> None:
    if args.list_devices:
        list_input_devices()
        return

    if args.list_system_devices:
        list_system_audio_devices()
        return

    live_fft_size = args.fft_size or DEFAULT_LIVE_FFT_SIZE
    measurement_fft_size = args.fft_size or DEFAULT_MEASUREMENT_FFT_SIZE

    if args.system_audio:
        if args.file is not None:
            raise ValueError("Im System-Audio-Modus keine Audiodatei angeben.")
        run_system_audio_mode(
            fft_size=live_fft_size,
            fps=args.fps,
            smoothing=args.smoothing,
            max_frequency_hz=args.max_frequency,
            waveform_ms=args.waveform_ms,
            system_device=args.system_device,
            sample_rate_hz=args.samplerate,
        )
        return

    if args.measure:
        if args.file is not None:
            raise ValueError("Im Einmessmodus keine Audiodatei angeben.")
        run_measurement_mode(
            device=args.device,
            sample_rate_hz=args.samplerate,
            channels=args.channels or 1,
            fft_size=measurement_fft_size,
            fps=args.fps,
            smoothing=args.smoothing,
            max_frequency_hz=args.max_frequency,
            rta_averaging=args.rta_averaging,
            delta_range_db=args.delta_range,
        )
        return

    if args.live:
        if args.file is not None:
            raise ValueError("Im Live-Modus keine Audiodatei angeben.")
        run_live_mode(
            device=args.device,
            sample_rate_hz=args.samplerate,
            channels=args.channels or 1,
            fft_size=live_fft_size,
            fps=args.fps,
            smoothing=args.smoothing,
            max_frequency_hz=args.max_frequency,
            waveform_ms=args.waveform_ms,
        )
        return

    run_file_mode(args)


def _run_argv(argv: list[str]) -> None:
    args = parse_arguments(argv)
    _validate_args(args)
    _dispatch(args)


def _ask_menu_file() -> str | None:
    raw = input("Pfad zur Audiodatei (leer = zurück): ").strip().strip('"').strip("'")
    return raw or None


def run_interactive_menu() -> None:
    """Interactive launcher used when the program is started without CLI arguments."""
    menu = f"""
Audio Analyzer {__version__}
{'=' * (15 + len(__version__))}

[1] Audiodatei analysieren (FFT)
[2] Spektrogramm einer Audiodatei
[3] Live-Audio / Mikrofon / Audiointerface
[4] System-Audio analysieren
[5] PA-Einmessmodus
[6] Audio-Eingabegeräte anzeigen
[7] System-Audio-Geräte anzeigen
[8] Kommandozeilen-Hilfe
[0] Beenden
"""

    commands = {
        "3": ["--live"],
        "4": ["--system-audio"],
        "5": ["--measure"],
        "6": ["--list-devices"],
        "7": ["--list-system-devices"],
    }

    while True:
        print(menu)
        try:
            choice = input("Auswahl: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBeendet.")
            return

        if choice == "0":
            return

        if choice == "8":
            print()
            build_parser().print_help()
            continue

        if choice in ("1", "2"):
            try:
                path = _ask_menu_file()
            except (EOFError, KeyboardInterrupt):
                print("\nZurück zum Menü.")
                continue
            if path is None:
                continue
            command = ["--", path] if choice == "1" else ["--spectrogram", "--", path]
        else:
            command = commands.get(choice)
            if command is None:
                print("\nUngültige Auswahl. Bitte 0 bis 8 eingeben.")
                continue

        try:
            _run_argv(command)
        except KeyboardInterrupt:
            print("\nVorgang abgebrochen.")
        except Exception as exc:
            # Im Doppelklick-Modus soll ein Fehler sichtbar bleiben und nicht sofort
            # zusammen mit dem Konsolenfenster verschwinden.
            print(f"\nFehler: {exc}")


def main(argv: list[str] | None = None) -> None:
    effective_argv = sys.argv[1:] if argv is None else list(argv)

    if not effective_argv:
        if sys.stdin.isatty():
            run_interactive_menu()
        else:
            # In Pipes/CI nicht auf Benutzereingabe warten.
            build_parser().print_help()
        return

    _run_argv(effective_argv)
