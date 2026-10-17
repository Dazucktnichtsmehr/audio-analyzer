# Standalone-Builds

Audio Analyzer kann als eigenständige ausführbare Datei verteilt werden. Auf dem Zielrechner ist dann **kein Python** nötig.

## Warum pro Betriebssystem separat bauen?

PyInstaller ist kein Cross-Compiler. Ein Windows-`.exe` muss unter Windows gebaut werden, ein macOS-Binary unter macOS und ein Linux-Binary unter Linux.

## Lokal bauen

Im Projektordner:

```bash
python -m pip install . pytest pyinstaller pyinstaller-hooks-contrib
python packaging/build_binary.py
```

Danach liegt unter `release/` ein Archiv für die aktuelle Plattform.

### Ausgaben

- Windows: `AudioAnalyzer-1.3.0-windows-x64.zip` mit `audio-analyzer.exe`
- macOS: `AudioAnalyzer-1.3.0-macos-arm64.tar.gz` bzw. `...-x64.tar.gz`
- Linux: `AudioAnalyzer-1.3.0-linux-x64.tar.gz`

Die Architektur im Dateinamen wird automatisch aus dem Build-Rechner ermittelt.

## Alle Plattformen automatisch mit GitHub Actions bauen

Die Datei `.github/workflows/build-binaries.yml` baut bei manuellem Start oder bei einem Git-Tag wie `v1.3.0` automatisch:

- Windows x64
- Linux x64
- macOS Apple Silicon (arm64)
- macOS Intel (x64)

Die fertigen Archive erscheinen anschließend unter **Actions → Workflow run → Artifacts**.

## Warum auf macOS kein `.app`?

Audio Analyzer bleibt bewusst ein CLI-Programm. Eine `.app` ist für ein Programm mit Kommandozeilenargumenten (`--live`, `--measure`, `--system-audio`, Dateipfade usw.) unpraktisch, weil beim Doppelklick kein Terminal für Ein-/Ausgabe und Argumente vorhanden ist. Der macOS-Build ist deshalb ein eigenständiges UNIX-Executable namens `audio-analyzer`.

Eine `.app` ist sinnvoll, sobald ein grafischer Launcher bzw. eine echte GUI hinzukommt.

## FFmpeg

FFmpeg wird nicht in das Binary eingebettet. Die direkte Audioanalyse funktioniert mit den von `libsndfile` unterstützten Formaten; für zusätzliche Container/Codecs muss FFmpeg separat im `PATH` vorhanden sein.

## Linux

Der Python-Interpreter und Python-Pakete werden gebündelt, aber Linux-Systemaudio hängt weiterhin vom Audio-Stack des Betriebssystems ab. Für Debian/Ubuntu sind typischerweise mindestens nötig:

```bash
sudo apt install libportaudio2 libpulse0
```

`--system-audio` benötigt PulseAudio oder PipeWire mit `pipewire-pulse`.

## Signierung

Die erzeugten Builds sind ohne weitere Konfiguration unsigniert:

- macOS kann deshalb Gatekeeper anzeigen.
- Windows kann deshalb SmartScreen anzeigen.

Für öffentliche Distribution sollten später Apple Developer ID + Notarisierung bzw. Windows Code Signing ergänzt werden.
