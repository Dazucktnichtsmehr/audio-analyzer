# Changelog

## 1.3.0

- Interaktives Startmenü beim Start ohne Argumente / per Doppelklick
- Direkter Zugriff auf Datei-FFT, Spektrogramm, Live-Audio, Systemaudio und PA-Einmessung
- Geräte- und Hilfsansichten direkt aus dem Startmenü
- CLI-Aufrufe mit `--live`, `--measure`, `--system-audio` usw. bleiben unverändert
- Nicht-interaktive Starts ohne Argumente blockieren nicht, sondern zeigen die Hilfe

## 1.2.0

- Standalone-Buildsystem mit PyInstaller
- GitHub-Actions-Matrix für Windows x64, Linux x64, macOS arm64 und macOS x64
- Release-Archive werden automatisch erzeugt
- Kein Python auf dem Zielrechner erforderlich
- BUILDING.md mit lokalen und CI-Build-Anweisungen

## 1.1.0

- Natives/cross-platform Systemaudio: macOS Core Audio Tap, Windows WASAPI Loopback, Linux PulseAudio/PipeWire-Pulse Monitor
