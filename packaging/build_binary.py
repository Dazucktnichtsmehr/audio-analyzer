from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "packaging" / "entrypoint.py"
DIST_DIR = PROJECT_ROOT / "dist-binary"
BUILD_DIR = PROJECT_ROOT / "build-pyinstaller"
SPEC_DIR = PROJECT_ROOT / "build-spec"
RELEASE_DIR = PROJECT_ROOT / "release"


def run(command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def normalized_arch() -> str:
    machine = platform.machine().lower()
    aliases = {
        "amd64": "x64",
        "x86_64": "x64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    return aliases.get(machine, machine.replace(" ", "-"))


def platform_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(f"Nicht unterstützte Build-Plattform: {sys.platform}")


def pyinstaller_args() -> list[str]:
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name",
        "audio-analyzer",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(SPEC_DIR),
        "--collect-all",
        "matplotlib",
        "--collect-all",
        "soundfile",
        "--collect-all",
        "sounddevice",
    ]

    if sys.platform == "darwin":
        args += ["--collect-all", "catap", "--hidden-import", "catap"]
    elif sys.platform == "win32":
        args += [
            "--collect-all",
            "pyaudiowpatch",
            "--hidden-import",
            "pyaudiowpatch",
        ]
    elif sys.platform.startswith("linux"):
        args += ["--collect-all", "soundcard", "--hidden-import", "soundcard"]

    args.append(str(ENTRYPOINT))
    return args


def write_binary_readme(folder: Path) -> None:
    system = platform_name()
    lines = [
        "Audio Analyzer 1.2.0",
        "====================",
        "",
        "Dieses Release enthält eine eigenständige Programmdatei.",
        "Python muss auf dem Zielrechner nicht installiert sein.",
        "",
        "Beispiele:",
        "  audio-analyzer --help",
        "  audio-analyzer --live",
        "  audio-analyzer --measure",
        "  audio-analyzer --system-audio",
        "  audio-analyzer song.mp3 --spectrogram",
        "",
        "FFmpeg ist optional und wird NICHT mitgeliefert. Für Formate, die",
        "libsndfile nicht direkt lesen kann, muss ffmpeg im PATH vorhanden sein.",
    ]
    if system == "linux":
        lines += [
            "",
            "Linux-Hinweis:",
            "  Live-/Systemaudio benötigt die Audio-Laufzeitbibliotheken des Systems,",
            "  insbesondere PortAudio sowie PulseAudio oder PipeWire mit pipewire-pulse.",
            "  Debian/Ubuntu typischerweise:",
            "    sudo apt install libportaudio2 libpulse0",
        ]
    elif system == "macos":
        lines += [
            "",
            "macOS-Hinweis:",
            "  Systemaudio benötigt macOS 14.2 oder neuer.",
            "  Ein nicht signierter Download kann von Gatekeeper bestätigt werden müssen.",
        ]
    elif system == "windows":
        lines += [
            "",
            "Windows-Hinweis:",
            "  Systemaudio verwendet WASAPI Loopback.",
            "  Bei einem nicht signierten Download kann Windows SmartScreen warnen.",
        ]
    (folder / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_archive(binary: Path) -> Path:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    system = platform_name()
    arch = normalized_arch()
    stem = f"AudioAnalyzer-1.2.0-{system}-{arch}"
    staging = RELEASE_DIR / stem
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    target_name = "audio-analyzer.exe" if system == "windows" else "audio-analyzer"
    shutil.copy2(binary, staging / target_name)
    if system != "windows":
        os.chmod(staging / target_name, 0o755)
    write_binary_readme(staging)

    if system == "windows":
        archive = RELEASE_DIR / f"{stem}.zip"
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in staging.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=Path(stem) / path.relative_to(staging))
    else:
        archive = RELEASE_DIR / f"{stem}.tar.gz"
        if archive.exists():
            archive.unlink()
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(staging, arcname=stem)

    print(f"Release erstellt: {archive}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone-Binary mit PyInstaller bauen")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="pytest vor dem Build überspringen",
    )
    args = parser.parse_args()

    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"])

    for path in (DIST_DIR, BUILD_DIR, SPEC_DIR):
        if path.exists():
            shutil.rmtree(path)

    run(pyinstaller_args())

    executable = DIST_DIR / ("audio-analyzer.exe" if sys.platform == "win32" else "audio-analyzer")
    if not executable.exists():
        raise RuntimeError(f"PyInstaller-Ausgabe fehlt: {executable}")

    run([str(executable), "--version"])
    run([str(executable), "--help"])
    build_archive(executable)


if __name__ == "__main__":
    main()
