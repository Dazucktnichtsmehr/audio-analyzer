from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .models import AudioData


def _validate_audio_request(
    path: Path,
    start_s: float,
    duration_s: float | None,
) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    if not path.is_file():
        raise ValueError(f"Pfad ist keine Datei: {path}")
    if start_s < 0:
        raise ValueError("start_s darf nicht negativ sein.")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s muss größer als 0 sein oder None.")
    return path


def _finish_audio_data(
    *,
    data: np.ndarray,
    sample_rate_hz: int,
    source_channels: int,
    start_s: float,
    source_path: Path,
    decoder: str,
) -> AudioData:
    if data.ndim == 1:
        samples = data.astype(np.float64, copy=False)
    elif data.ndim == 2:
        samples = np.mean(data, axis=1, dtype=np.float64)
    else:
        raise ValueError(f"Unerwartete Audio-Datenform: {data.shape}")

    if samples.size < 4:
        raise ValueError("Der gewählte Audioausschnitt ist zu kurz für eine FFT.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Die Audiodatei enthält ungültige Samplewerte (NaN/Inf).")

    return AudioData(
        samples=samples,
        sample_rate_hz=int(sample_rate_hz),
        start_s=float(start_s),
        duration_s=float(samples.size / sample_rate_hz),
        source_path=source_path,
        source_channels=int(source_channels),
        decoder=decoder,
    )


def _load_with_soundfile(
    path: Path,
    start_s: float,
    duration_s: float | None,
) -> AudioData:
    with sf.SoundFile(path, mode="r") as audio_file:
        sample_rate_hz = int(audio_file.samplerate)
        source_channels = int(audio_file.channels)
        total_frames = int(len(audio_file))

        if sample_rate_hz <= 0 or total_frames <= 0:
            raise ValueError("Die Audiodatei enthält keine verwertbaren Samples.")

        total_duration_s = total_frames / sample_rate_hz
        start_frame = int(round(start_s * sample_rate_hz))
        if start_frame >= total_frames:
            raise ValueError(
                f"Startzeit {start_s:.3f} s liegt außerhalb der Datei "
                f"({total_duration_s:.3f} s)."
            )

        if duration_s is None:
            frames_to_read = total_frames - start_frame
        else:
            frames_to_read = min(
                int(round(duration_s * sample_rate_hz)),
                total_frames - start_frame,
            )

        audio_file.seek(start_frame)
        data = audio_file.read(
            frames=frames_to_read,
            dtype="float64",
            always_2d=True,
        )

    return _finish_audio_data(
        data=data,
        sample_rate_hz=sample_rate_hz,
        source_channels=source_channels,
        start_s=start_s,
        source_path=path,
        decoder="soundfile/libsndfile",
    )


def _ffmpeg_tools_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_with_ffprobe(path: Path) -> tuple[int, int]:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "ffprobe konnte keinen Audiostream lesen."
            + (f"\nffprobe: {error}" if error else "")
        )

    try:
        payload = json.loads(result.stdout.decode("utf-8"))
        stream = payload.get("streams", [])[0]
        sample_rate_hz = int(stream["sample_rate"])
        source_channels = int(stream["channels"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("ffprobe lieferte keine verwertbaren Audio-Metadaten.") from exc

    if sample_rate_hz <= 0 or source_channels <= 0:
        raise RuntimeError("Ungültige Audio-Metadaten von ffprobe.")
    return sample_rate_hz, source_channels


def _load_with_ffmpeg(
    path: Path,
    start_s: float,
    duration_s: float | None,
) -> AudioData:
    if not _ffmpeg_tools_available():
        raise RuntimeError(
            "FFmpeg/ffprobe wurde nicht gefunden. Installiere FFmpeg und stelle "
            "sicher, dass ffmpeg und ffprobe im PATH liegen."
        )

    sample_rate_hz, source_channels = _probe_with_ffprobe(path)
    command = ["ffmpeg", "-v", "error"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.9f}"]
    command += ["-i", str(path), "-map", "0:a:0", "-vn"]
    if duration_s is not None:
        command += ["-t", f"{duration_s:.9f}"]
    command += [
        "-ac", "1",
        "-ar", str(sample_rate_hz),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "FFmpeg konnte die Audiodatei nicht dekodieren."
            + (f"\nffmpeg: {error}" if error else "")
        )

    data = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if data.size == 0:
        raise ValueError(
            "FFmpeg lieferte keine Samples. Liegt die Startzeit eventuell "
            "außerhalb der Datei?"
        )

    return _finish_audio_data(
        data=data,
        sample_rate_hz=sample_rate_hz,
        source_channels=source_channels,
        start_s=start_s,
        source_path=path,
        decoder="FFmpeg",
    )


def load_audio_mono(
    path: Path,
    start_s: float = 0.0,
    duration_s: float | None = 5.0,
    decoder: str = "auto",
) -> AudioData:
    path = _validate_audio_request(path, start_s, duration_s)
    if decoder not in {"auto", "soundfile", "ffmpeg"}:
        raise ValueError("decoder muss 'auto', 'soundfile' oder 'ffmpeg' sein.")

    if decoder in {"auto", "soundfile"}:
        try:
            return _load_with_soundfile(path, start_s, duration_s)
        except sf.LibsndfileError:
            if decoder == "soundfile":
                raise RuntimeError("soundfile/libsndfile unterstützt diese Datei nicht.")

    if decoder in {"auto", "ffmpeg"}:
        return _load_with_ffmpeg(path, start_s, duration_s)

    raise RuntimeError("Die Audiodatei konnte nicht dekodiert werden.")
