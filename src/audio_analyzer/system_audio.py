from __future__ import annotations

import sys
import threading
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .buffer import RingBuffer
from .constants import DB_FLOOR, DEFAULT_MIN_LOG_FREQUENCY_HZ
from .dsp import calculate_spectrum_from_samples
from .plotting import configure_log_audio_axis


LINUX_DEFAULT_SAMPLE_RATE_HZ = 48_000
SOUNDCARD_RECORD_FRAMES = 2_048
SOUNDCARD_BLOCKSIZE = 4_096
WINDOWS_WASAPI_CHUNK_FRAMES = 2_048


def _validate_common_args(
    *,
    fft_size: int,
    fps: float,
    smoothing: float,
    waveform_ms: float,
) -> None:
    if fft_size < 256:
        raise ValueError("--fft-size muss mindestens 256 sein.")
    if fps <= 0 or fps > 120:
        raise ValueError("--fps muss > 0 und <= 120 sein.")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("--smoothing muss im Bereich (0, 1] liegen.")
    if waveform_ms <= 0:
        raise ValueError("--waveform-ms muss größer als 0 sein.")


def _import_catap():
    try:
        from catap import record_system_audio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Für den nativen macOS-System-Audio-Modus fehlt 'catap'. "
            "Installieren mit: pip install 'catap>=0.6.0'"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "Der native System-Audio-Modus benötigt macOS 14.2 oder neuer. "
            f"catap konnte nicht geladen werden: {exc}"
        ) from exc
    return record_system_audio


def _import_pyaudiowpatch():
    try:
        import pyaudiowpatch as pyaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Für Systemaudio unter Windows fehlt 'PyAudioWPatch'. "
            "Installieren mit: pip install 'PyAudioWPatch>=0.2.12.9'"
        ) from exc
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"PyAudioWPatch konnte nicht geladen werden: {exc}") from exc
    return pyaudio


def _import_soundcard():
    try:
        import soundcard as sc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Für Systemaudio unter Linux fehlt 'SoundCard'. "
            "Installieren mit: pip install 'SoundCard>=0.4.6'"
        ) from exc
    except (ImportError, OSError) as exc:
        if sys.platform.startswith("linux"):
            raise RuntimeError(
                "SoundCard konnte das Linux-Audiosystem nicht laden. "
                "Benötigt wird PulseAudio oder PipeWire mit PulseAudio-Kompatibilität "
                "(pipewire-pulse/libpulse)."
            ) from exc
        raise RuntimeError(f"SoundCard konnte nicht geladen werden: {exc}") from exc
    return sc


def _decode_signed_int24_le(data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size % 3 != 0:
        raise ValueError("Ungültige 24-Bit-PCM-Pufferlänge.")
    triples = raw.reshape(-1, 3).astype(np.int32, copy=False)
    values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    values = (values ^ 0x800000) - 0x800000
    return values.astype(np.float64) / 8_388_608.0


def _catap_audio_buffer_to_mono(buffer) -> np.ndarray:
    fmt = buffer.format
    channels = int(fmt.num_channels)
    bits = int(fmt.bits_per_sample)
    if channels < 1:
        raise ValueError("Core Audio meldet keine gültigen Audiokanäle.")

    if fmt.sample_type == "float":
        if bits != 32:
            raise ValueError(f"Nicht unterstütztes Core-Audio-Floatformat: {bits} Bit.")
        samples = np.frombuffer(buffer.data, dtype="<f4").astype(np.float64, copy=False)
    elif fmt.sample_type == "signed_integer":
        if bits == 16:
            samples = np.frombuffer(buffer.data, dtype="<i2").astype(np.float64) / 32_768.0
        elif bits == 24:
            samples = _decode_signed_int24_le(buffer.data)
        elif bits == 32:
            samples = (
                np.frombuffer(buffer.data, dtype="<i4").astype(np.float64)
                / 2_147_483_648.0
            )
        else:
            raise ValueError(f"Nicht unterstütztes Core-Audio-Integerformat: {bits} Bit.")
    else:
        raise ValueError(f"Nicht unterstützter Core-Audio-Sampletyp: {fmt.sample_type!r}.")

    expected_samples = int(buffer.frame_count) * channels
    if samples.size != expected_samples:
        raise ValueError(
            "Core-Audio-Puffergröße passt nicht zu Frame-/Kanalzahl: "
            f"{samples.size} Samples statt {expected_samples}."
        )
    if channels == 1:
        return samples
    return np.mean(
        samples.reshape(int(buffer.frame_count), channels), axis=1, dtype=np.float64
    )


def _frames_to_mono(frames: Any) -> np.ndarray:
    """SoundCard-Frames (frames x channels) robust in Mono-Float64 umwandeln."""
    array = np.asarray(frames, dtype=np.float64)
    if array.size == 0:
        return np.empty(0, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1)
    if array.ndim != 2:
        raise ValueError(f"Unerwartete Systemaudio-Datenform: {array.shape}")
    if array.shape[1] < 1:
        return np.empty(0, dtype=np.float64)
    return np.mean(array, axis=1, dtype=np.float64)


def _run_live_spectrum_display(
    *,
    ring: RingBuffer,
    sample_rate_hz: int,
    fft_size: int,
    fps: float,
    smoothing: float,
    max_frequency_hz: float,
    waveform_ms: float,
    title: str,
    wave_title: str,
    spectrum_title: str,
    health_check=None,
) -> None:
    if sample_rate_hz <= 0:
        raise ValueError("Systemaudio meldet eine ungültige Samplerate.")

    display_max_hz = min(max_frequency_hz, sample_rate_hz / 2.0)
    if display_max_hz <= DEFAULT_MIN_LOG_FREQUENCY_HZ:
        raise ValueError("Samplerate ist zu niedrig für die Audioanzeige.")

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
    (wave_line,) = ax_wave.plot(
        waveform_time_ms, np.zeros(waveform_samples), linewidth=0.9
    )
    ax_wave.set_title(wave_title)
    ax_wave.set_xlabel("Zeit relativ zu jetzt [ms]")
    ax_wave.set_ylabel("Amplitude [FS]")
    ax_wave.set_xlim(waveform_time_ms[0], 0)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.grid(True, alpha=0.3)

    (spectrum_line,) = ax_spectrum.plot(
        plot_frequencies,
        np.full(plot_frequencies.size, DB_FLOOR),
        linewidth=1.0,
    )
    ax_spectrum.set_title(spectrum_title)
    ax_spectrum.set_xlabel("Frequenz [Hz]")
    ax_spectrum.set_ylabel("Pegel [dBFS]")
    ax_spectrum.set_ylim(DB_FLOOR, 3.0)
    configure_log_audio_axis(
        ax_spectrum, DEFAULT_MIN_LOG_FREQUENCY_HZ, display_max_hz
    )
    info_text = ax_spectrum.text(
        0.99, 0.97, "", transform=ax_spectrum.transAxes, ha="right", va="top"
    )
    fig.suptitle(title)
    fig.tight_layout()

    smoothed_power: np.ndarray | None = None
    floor_power = (10.0 ** (DB_FLOOR / 20.0)) ** 2

    plt.show(block=False)
    update_interval_s = 1.0 / fps
    next_update = time.monotonic()

    while plt.fignum_exists(fig.number):
        if health_check is not None:
            health_check()

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


def _run_macos_system_audio(
    *,
    fft_size: int,
    fps: float,
    smoothing: float,
    max_frequency_hz: float,
    waveform_ms: float,
) -> None:
    record_system_audio = _import_catap()
    ring = RingBuffer(fft_size)
    callback_error: list[BaseException] = []
    callback_error_lock = threading.Lock()

    def on_buffer(buffer) -> None:
        try:
            ring.write(_catap_audio_buffer_to_mono(buffer))
        except BaseException as exc:
            with callback_error_lock:
                if not callback_error:
                    callback_error.append(exc)

    session = record_system_audio(on_buffer=on_buffer, mono=True)
    print("\n--- macOS-System-Audio (native Core Audio Tap) ---")
    print("Kein virtuelles Audiogerät und kein Audio-Routing erforderlich.")
    print("Beim ersten Start kann macOS nach Systemaudio-Aufzeichnung fragen.")

    session_started = False
    try:
        session.start()
        session_started = True
        stream_format = session.stream_format
        if stream_format is None:
            raise RuntimeError("Core Audio hat nach dem Start kein Streamformat geliefert.")

        sample_rate_hz = int(round(float(stream_format.sample_rate)))
        source_channels = int(stream_format.num_channels)
        bits_per_sample = int(stream_format.bits_per_sample)
        sample_type = str(stream_format.sample_type)

        print(f"Samplerate:            {sample_rate_hz} Hz")
        print(f"Tap-Kanäle:            {source_channels}")
        print(f"PCM-Format:            {sample_type}, {bits_per_sample} Bit")
        print(f"FFT-Größe:             {fft_size}")
        print(f"Frequenzauflösung:     {sample_rate_hz / fft_size:.3f} Hz")
        print(f"FFT-Zeitfenster:       {fft_size / sample_rate_hz * 1000.0:.1f} ms")
        print(f"Anzeige-Updates:       {fps:.1f} FPS")
        print(f"Glättung:              {smoothing:.2f}")
        print("Beenden: Diagramm schließen oder Ctrl+C\n")

        def health_check() -> None:
            with callback_error_lock:
                pending_error = callback_error[0] if callback_error else None
            if pending_error is not None:
                raise RuntimeError(
                    "Fehler beim Verarbeiten eines Core-Audio-Puffers."
                ) from pending_error
            if getattr(session, "capture_failed", False):
                raise RuntimeError(
                    "Der Core-Audio-Systemtap ist während der Aufnahme fehlgeschlagen."
                )

        _run_live_spectrum_display(
            ring=ring,
            sample_rate_hz=sample_rate_hz,
            fft_size=fft_size,
            fps=fps,
            smoothing=smoothing,
            max_frequency_hz=max_frequency_hz,
            waveform_ms=waveform_ms,
            title=f"Native macOS System Audio – {sample_rate_hz} Hz – Core Audio Tap",
            wave_title="Systemaudio – Live-Zeitsignal",
            spectrum_title="Systemaudio – Live-Spektrum",
            health_check=health_check,
        )
    except KeyboardInterrupt:
        print("\nSystem-Audio-Modus beendet.")
    finally:
        if session_started:
            try:
                session.stop()
            finally:
                if getattr(session, "captured_only_silence", False):
                    print(
                        "\nHinweis: Der Core-Audio-Tap hat nur Stille geliefert. "
                        "Falls Ton lief, prüfe die macOS-Berechtigung für "
                        "Systemaudio-Aufzeichnung und starte das Terminal neu."
                    )


def _resolve_soundcard_loopback(sc, device: str | None):
    speaker = sc.default_speaker() if device is None else sc.get_speaker(device)

    # Windows erzeugt die Loopback-Mikrofone direkt aus den WASAPI-Ausgängen.
    # Unter PulseAudio/PipeWire-Pulse heißen sie typischerweise "Monitor of ...".
    # Der Speaker-Name funktioniert daher auf beiden Backends als guter Matcher.
    try:
        microphone = sc.get_microphone(str(speaker.name), include_loopback=True)
    except (IndexError, RuntimeError):
        candidates = [
            mic
            for mic in sc.all_microphones(include_loopback=True)
            if bool(getattr(mic, "isloopback", False))
        ]
        if len(candidates) == 1:
            microphone = candidates[0]
        else:
            names = "\n".join(f"  - {mic.name}" for mic in candidates) or "  (keine)"
            raise RuntimeError(
                f"Kein eindeutiger Loopback-Eingang für '{speaker.name}' gefunden.\n"
                "Verfügbare Loopback-/Monitor-Quellen:\n"
                f"{names}\n"
                "Nutze --system-device <Ausgangsname>, um einen Ausgang explizit zu wählen."
            )

    return speaker, microphone


def _resolve_windows_loopback_device(p, system_device: str | None):
    if system_device is None:
        return p.get_default_wasapi_loopback()

    # Numerischer Index: Loopback direkt nutzen oder den zugehörigen
    # WASAPI-Loopback des normalen Ausgabegerätes suchen.
    try:
        index = int(system_device)
    except ValueError:
        index = None

    if index is not None:
        info = p.get_device_info_by_index(index)
        if bool(info.get("isLoopbackDevice", False)):
            return info
        if int(info.get("maxOutputChannels", 0)) > 0:
            return p.get_wasapi_loopback_analogue_by_index(index)
        raise ValueError(
            f"Windows-Audiogerät {index} ist weder Loopback noch Ausgangsgerät."
        )

    query = system_device.casefold()
    candidates = list(p.get_loopback_device_info_generator())
    exact = [d for d in candidates if str(d.get("name", "")).casefold() == query]
    if len(exact) == 1:
        return exact[0]

    matches = [d for d in candidates if query in str(d.get("name", "")).casefold()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Kein WASAPI-Loopback passend zu '{system_device}' gefunden.")

    names = "\n".join(f"  - {d['name']} (Index {d['index']})" for d in matches)
    raise ValueError(
        f"Mehrere WASAPI-Loopbacks passen zu '{system_device}':\n{names}\n"
        "Bitte Name genauer angeben oder Index verwenden."
    )


def list_system_audio_devices() -> None:
    """Listet wählbare Systemaudio-Ausgänge für das aktuelle Betriebssystem."""
    if sys.platform == "darwin":
        print("\n--- Systemaudio ---")
        print(
            "macOS nutzt einen Core Audio Process Tap auf den Systemmix. "
            "Ein separates Ausgabegerät muss nicht gewählt werden."
        )
        return

    if sys.platform == "win32":
        pyaudio = _import_pyaudiowpatch()
        p = pyaudio.PyAudio()
        try:
            default = p.get_default_wasapi_loopback()
            default_index = int(default["index"])
            print("\n--- Windows WASAPI-Loopbacks ---")
            for device in p.get_loopback_device_info_generator():
                marker = " [Standard]" if int(device["index"]) == default_index else ""
                print(
                    f"{int(device['index']):>3}: {device['name']} "
                    f"| Inputs: {int(device['maxInputChannels'])} "
                    f"| SR: {float(device['defaultSampleRate']):.0f} Hz{marker}"
                )
        finally:
            p.terminate()
        return

    if sys.platform.startswith("linux"):
        sc = _import_soundcard()
        default_speaker = sc.default_speaker()
        print("\n--- Linux Systemaudio-Ausgänge ---")
        for index, speaker in enumerate(sc.all_speakers()):
            default_marker = " [Standard]" if speaker.id == default_speaker.id else ""
            print(
                f"{index:>3}: {speaker.name} | Kanäle: {int(speaker.channels)}{default_marker}"
            )
        return

    raise RuntimeError(
        f"Systemaudio wird auf dieser Plattform noch nicht unterstützt: {sys.platform}"
    )


def _run_windows_wasapi_system_audio(
    *,
    fft_size: int,
    fps: float,
    smoothing: float,
    max_frequency_hz: float,
    waveform_ms: float,
    system_device: str | None,
    sample_rate_hz: int | None,
) -> None:
    pyaudio = _import_pyaudiowpatch()
    p = pyaudio.PyAudio()
    stop_event = threading.Event()
    capture_error: list[BaseException] = []
    capture_error_lock = threading.Lock()
    received_audio = threading.Event()

    try:
        device = _resolve_windows_loopback_device(p, system_device)
        device_index = int(device["index"])
        channels = max(1, int(device["maxInputChannels"]))
        native_sample_rate = int(round(float(device["defaultSampleRate"])))
        if sample_rate_hz is None:
            sample_rate_hz = native_sample_rate
        if sample_rate_hz <= 0:
            raise ValueError("--samplerate muss größer als 0 sein.")

        ring = RingBuffer(fft_size)

        def capture_worker() -> None:
            stream = None
            try:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sample_rate_hz,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=WINDOWS_WASAPI_CHUNK_FRAMES,
                )
                while not stop_event.is_set():
                    raw = stream.read(
                        WINDOWS_WASAPI_CHUNK_FRAMES,
                        exception_on_overflow=False,
                    )
                    samples = np.frombuffer(raw, dtype="<f4")
                    expected_multiple = channels
                    if samples.size == 0:
                        continue
                    if samples.size % expected_multiple != 0:
                        raise RuntimeError(
                            "WASAPI lieferte einen Puffer, dessen Samplezahl nicht "
                            "zur Kanalzahl passt."
                        )
                    if channels > 1:
                        mono = np.mean(
                            samples.reshape(-1, channels),
                            axis=1,
                            dtype=np.float64,
                        )
                    else:
                        mono = samples.astype(np.float64, copy=False)
                    ring.write(mono)
                    received_audio.set()
            except BaseException as exc:
                with capture_error_lock:
                    if not capture_error:
                        capture_error.append(exc)
            finally:
                if stream is not None:
                    try:
                        if stream.is_active():
                            stream.stop_stream()
                    finally:
                        stream.close()

        print("\n--- Windows-System-Audio (native WASAPI Loopback) ---")
        print(f"Loopback:              {device['name']}")
        print(f"Samplerate:            {sample_rate_hz} Hz")
        if sample_rate_hz != native_sample_rate:
            print(f"Native Geräte-SR:      {native_sample_rate} Hz")
        print(f"Aufnahmekanäle:        {channels}")
        print(f"FFT-Größe:             {fft_size}")
        print(f"Frequenzauflösung:     {sample_rate_hz / fft_size:.3f} Hz")
        print(f"FFT-Zeitfenster:       {fft_size / sample_rate_hz * 1000.0:.1f} ms")
        print(f"Anzeige-Updates:       {fps:.1f} FPS")
        print(f"Glättung:              {smoothing:.2f}")
        print("Kein virtuelles Audiokabel erforderlich.")
        print("Beenden: Diagramm schließen oder Ctrl+C\n")

        thread = threading.Thread(
            target=capture_worker,
            name="audio-analyzer-wasapi-capture",
            daemon=True,
        )
        thread.start()
        start_time = time.monotonic()

        def health_check() -> None:
            with capture_error_lock:
                pending_error = capture_error[0] if capture_error else None
            if pending_error is not None:
                raise RuntimeError(
                    "WASAPI Loopback konnte den Systemton nicht weiter aufnehmen."
                ) from pending_error
            if (
                not received_audio.is_set()
                and time.monotonic() - start_time > 5.0
                and not getattr(health_check, "warned", False)
            ):
                print(
                    "Hinweis: Noch keine WASAPI-Audioblöcke empfangen. "
                    "Starte testweise eine Wiedergabe und prüfe ggf. --system-device."
                )
                health_check.warned = True

        try:
            _run_live_spectrum_display(
                ring=ring,
                sample_rate_hz=sample_rate_hz,
                fft_size=fft_size,
                fps=fps,
                smoothing=smoothing,
                max_frequency_hz=max_frequency_hz,
                waveform_ms=waveform_ms,
                title=f"Windows WASAPI Loopback – {device['name']} – {sample_rate_hz} Hz",
                wave_title="Systemaudio – Live-Zeitsignal",
                spectrum_title="Systemaudio – Live-Spektrum",
                health_check=health_check,
            )
        except KeyboardInterrupt:
            print("\nSystem-Audio-Modus beendet.")
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
    finally:
        p.terminate()


def _run_linux_system_audio(
    *,
    fft_size: int,
    fps: float,
    smoothing: float,
    max_frequency_hz: float,
    waveform_ms: float,
    system_device: str | None,
    sample_rate_hz: int | None,
) -> None:
    sc = _import_soundcard()
    speaker, microphone = _resolve_soundcard_loopback(sc, system_device)

    if sample_rate_hz is None:
        sample_rate_hz = LINUX_DEFAULT_SAMPLE_RATE_HZ
    if sample_rate_hz <= 0:
        raise ValueError("--samplerate muss größer als 0 sein.")

    speaker_channels = max(1, int(getattr(speaker, "channels", 2)))
    capture_channels = min(speaker_channels, 8)

    ring = RingBuffer(fft_size)
    stop_event = threading.Event()
    capture_error: list[BaseException] = []
    capture_error_lock = threading.Lock()
    received_audio = threading.Event()

    def capture_worker() -> None:
        try:
            with microphone.recorder(
                samplerate=sample_rate_hz,
                channels=capture_channels,
                blocksize=SOUNDCARD_BLOCKSIZE,
            ) as recorder:
                while not stop_event.is_set():
                    frames = recorder.record(numframes=SOUNDCARD_RECORD_FRAMES)
                    mono = _frames_to_mono(frames)
                    if mono.size == 0:
                        time.sleep(0.01)
                        continue
                    ring.write(mono)
                    received_audio.set()
        except BaseException as exc:
            with capture_error_lock:
                if not capture_error:
                    capture_error.append(exc)

    print("\n--- Linux-System-Audio (PulseAudio / PipeWire-Pulse Monitor) ---")
    print(f"Ausgang:               {speaker.name}")
    print(f"Monitor:               {microphone.name}")
    print(f"Samplerate:            {sample_rate_hz} Hz")
    print(f"Aufnahmekanäle:        {capture_channels}")
    print(f"FFT-Größe:             {fft_size}")
    print(f"Frequenzauflösung:     {sample_rate_hz / fft_size:.3f} Hz")
    print(f"FFT-Zeitfenster:       {fft_size / sample_rate_hz * 1000.0:.1f} ms")
    print(f"Anzeige-Updates:       {fps:.1f} FPS")
    print(f"Glättung:              {smoothing:.2f}")
    print(
        "Benötigt PulseAudio oder PipeWire mit PulseAudio-Kompatibilität "
        "(pipewire-pulse); kein virtuelles Audiokabel erforderlich."
    )
    print("Beenden: Diagramm schließen oder Ctrl+C\n")

    thread = threading.Thread(
        target=capture_worker,
        name="audio-analyzer-linux-monitor-capture",
        daemon=True,
    )
    thread.start()
    start_time = time.monotonic()

    def health_check() -> None:
        with capture_error_lock:
            pending_error = capture_error[0] if capture_error else None
        if pending_error is not None:
            raise RuntimeError(
                "Der Linux-Monitor konnte den Systemton nicht weiter aufnehmen."
            ) from pending_error
        if (
            not received_audio.is_set()
            and time.monotonic() - start_time > 5.0
            and not getattr(health_check, "warned", False)
        ):
            print(
                "Hinweis: Noch keine Monitor-Audioblöcke empfangen. "
                "Starte testweise eine Wiedergabe und prüfe ggf. --system-device."
            )
            health_check.warned = True

    try:
        _run_live_spectrum_display(
            ring=ring,
            sample_rate_hz=sample_rate_hz,
            fft_size=fft_size,
            fps=fps,
            smoothing=smoothing,
            max_frequency_hz=max_frequency_hz,
            waveform_ms=waveform_ms,
            title=f"Linux System Audio – {speaker.name} – {sample_rate_hz} Hz",
            wave_title="Systemaudio – Live-Zeitsignal",
            spectrum_title="Systemaudio – Live-Spektrum",
            health_check=health_check,
        )
    except KeyboardInterrupt:
        print("\nSystem-Audio-Modus beendet.")
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            print(
                "Hinweis: Der Linux-Systemaudio-Thread beendet sich verzögert; "
                "der Prozess wird beim Schließen trotzdem freigegeben."
            )


def run_system_audio_mode(
    *,
    fft_size: int,
    fps: float,
    smoothing: float,
    max_frequency_hz: float,
    waveform_ms: float,
    system_device: str | None = None,
    sample_rate_hz: int | None = None,
) -> None:
    """Plattformgerechte Systemaudio-Aufnahme ohne virtuelle Audiokabel.

    macOS 14.2+: Core Audio Process Taps via catap
    Windows:     native WASAPI Loopback via PyAudioWPatch
    Linux:       PulseAudio-Monitor bzw. PipeWire-Pulse-Monitor via SoundCard
    """
    _validate_common_args(
        fft_size=fft_size,
        fps=fps,
        smoothing=smoothing,
        waveform_ms=waveform_ms,
    )

    if sys.platform == "darwin":
        if system_device is not None:
            raise ValueError(
                "--system-device wird unter macOS nicht benötigt; "
                "Core Audio tappt den Systemmix direkt."
            )
        if sample_rate_hz is not None:
            raise ValueError(
                "--samplerate kann bei macOS-Systemaudio nicht erzwungen werden; "
                "das native Core-Audio-Tapformat wird verwendet."
            )
        _run_macos_system_audio(
            fft_size=fft_size,
            fps=fps,
            smoothing=smoothing,
            max_frequency_hz=max_frequency_hz,
            waveform_ms=waveform_ms,
        )
        return

    if sys.platform == "win32":
        _run_windows_wasapi_system_audio(
            fft_size=fft_size,
            fps=fps,
            smoothing=smoothing,
            max_frequency_hz=max_frequency_hz,
            waveform_ms=waveform_ms,
            system_device=system_device,
            sample_rate_hz=sample_rate_hz,
        )
        return

    if sys.platform.startswith("linux"):
        _run_linux_system_audio(
            fft_size=fft_size,
            fps=fps,
            smoothing=smoothing,
            max_frequency_hz=max_frequency_hz,
            waveform_ms=waveform_ms,
            system_device=system_device,
            sample_rate_hz=sample_rate_hz,
        )
        return

    raise RuntimeError(
        f"Systemaudio wird auf dieser Plattform noch nicht unterstützt: {sys.platform}"
    )
