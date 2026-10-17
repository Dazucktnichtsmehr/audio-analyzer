# Audio Analyzer 1.3

Kommandozeilen-Audioanalyse mit grafischer Darstellung.

## Funktionen

- Audiodateien: FFT, lineares Spektrum und dBFS-Spektrum
- STFT-Spektrogramm
- Live-Mikrofon / Audiointerface
- PA-Einmessmodus mit dauerhaft sichtbarem Live-Spektrum
- 1/3-Oktav-RTA, Fast/Slow Averaging, Peak Hold und Referenzvergleich
- Systemaudio ohne virtuelles Audiokabel:
  - macOS 14.2+: Apple Core Audio Process Taps
  - Windows 10/11: WASAPI Loopback
  - Linux: PulseAudio-Monitor bzw. PipeWire über `pipewire-pulse`
- SoundFile/libsndfile mit FFmpeg-Fallback für weitere Dateiformate

## Systeme

| Funktion | macOS 14.2+ | Windows 10/11 | Linux Desktop |
| --- | --- | --- | --- |
| Datei-Analyse | ✅ | ✅ | ✅ |
| Spektrogramm | ✅ | ✅ | ✅ |
| Live-Eingang | ✅ | ✅ | ✅ |
| PA-Einmessung | ✅ | ✅ | ✅ |
| Systemaudio | ✅ Core Audio Tap | ✅ WASAPI Loopback | ✅ PulseAudio/PipeWire-Pulse Monitor |

Unter Linux wird für Systemaudio PulseAudio oder die übliche PulseAudio-Kompatibilität von PipeWire (`pipewire-pulse`) benötigt. Es ist kein zusätzliches virtuelles Audiokabel erforderlich.


## Standalone-Version ohne Python

Neben der Python-/pipx-Installation kann Audio Analyzer mit PyInstaller als eigenständige Programmdatei gebaut werden. Auf dem Zielrechner ist dann kein Python erforderlich.

Wird das Programm ohne Kommandozeilenargumente gestartet (z. B. per Doppelklick auf `audio-analyzer.exe` unter Windows), öffnet sich ein interaktives Startmenü. Die bisherigen CLI-Befehle bleiben unverändert verfügbar.

- Windows: `audio-analyzer.exe`
- macOS: eigenständiges `audio-analyzer`-Executable für Terminal/CLI
- Linux: eigenständiges `audio-analyzer`-Executable

Ein GitHub-Actions-Workflow im Projekt baut automatisch Windows x64, Linux x64 sowie macOS arm64 und x64. Details stehen in `BUILDING.md`.

## Installation als CLI-Befehl

Python 3.11 oder neuer wird benötigt.

### Empfohlen: pipx

Im entpackten Projektordner:

```bash
pipx install .
```

Danach steht systemweit zur Verfügung:

```bash
audio-analyzer --help
audio-analyzer --version
```

Für Updates aus demselben Projektordner:

```bash
pipx install . --force
```

### Alternativ in einer virtuellen Umgebung

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install .
```

Die plattformspezifischen Systemaudio-Abhängigkeiten werden automatisch gewählt:

- macOS: `catap`
- Windows: `PyAudioWPatch`
- Linux: `SoundCard`

## FFmpeg

WAV/FLAC und weitere von libsndfile unterstützte Formate funktionieren direkt. Für möglichst breite Unterstützung, z. B. bei bestimmten MP3/AAC/M4A-Dateien, sollte FFmpeg installiert und im PATH verfügbar sein.

macOS mit Homebrew:

```bash
brew install ffmpeg
```

Unter Windows kann FFmpeg z. B. über `winget` installiert werden, unter Linux über den Paketmanager der Distribution.

## Beispiele

Datei analysieren:

```bash
audio-analyzer song.mp3
```

20 Sekunden ab Sekunde 60:

```bash
audio-analyzer song.mp3 --start 60 --duration 20
```

Spektrogramm:

```bash
audio-analyzer song.mp3 --spectrogram
```

Eingabegeräte anzeigen:

```bash
audio-analyzer --list-devices
```

Live-Mikrofon / Interface:

```bash
audio-analyzer --live
```

Bestimmtes Eingabegerät:

```bash
audio-analyzer --live --device 2
```

PA-Einmessmodus:

```bash
audio-analyzer --measure
```

Schnelleres RTA-Averaging:

```bash
audio-analyzer --measure --rta-averaging fast
```

### Systemaudio

Auf allen unterstützten Systemen ist der Standardbefehl gleich:

```bash
audio-analyzer --system-audio
```

Unter Windows/Linux kann man die verfügbaren Ausgänge anzeigen:

```bash
audio-analyzer --list-system-devices
```

und einen bestimmten Ausgang wählen:

```bash
audio-analyzer --system-audio --system-device "Lautsprecher"
```

Optional kann unter Windows/Linux eine andere Capture-Samplerate gewählt werden:

```bash
audio-analyzer --system-audio --samplerate 44100
```

Ohne Angabe werden 48 kHz verwendet. Unter macOS wird die native Samplerate des Core-Audio-Taps übernommen.

## Systemaudio-Backends

### macOS

Ab macOS 14.2 werden Apple Core Audio Process Taps genutzt. Kein BlackHole, kein Multi-Output-Device und kein manuelles Routing. Beim ersten Start kann macOS nach der Berechtigung für Systemaudio-Aufzeichnung fragen.

### Windows

Windows verwendet WASAPI Loopback über PyAudioWPatch auf dem aktuellen oder gewählten Audioausgang. Der Ton wird weiterhin normal wiedergegeben. Standardmäßig wird die native Samplerate des WASAPI-Loopbacks verwendet; optional kann `--samplerate` gesetzt werden.

### Linux

Linux verwendet die Monitor-Quelle des aktuellen bzw. gewählten Ausgangs über PulseAudio. Auf heutigen PipeWire-Desktops funktioniert dies über `pipewire-pulse`. Falls eine minimalistische Linux-Installation nur reines ALSA oder PipeWire ohne PulseAudio-Kompatibilität verwendet, ist `--system-audio` in dieser Version nicht verfügbar; Datei-, Live- und Einmessmodus funktionieren davon unabhängig.

## PA-Einmessung

Der Einmessmodus zeigt gleichzeitig:

1. hochauflösendes Live-FFT-Spektrum mit Peak Hold,
2. 1/3-Oktav-RTA,
3. Differenz zu einer per Button gesetzten Referenz.

Für Frequenzgangmessungen ist Pink Noise sinnvoll. Die absolute Anzeige ist dBFS, nicht kalibriertes dB SPL. Relative Pegeländerungen in dB bleiben bei unveränderter Messkette aussagekräftig.

## Entwicklung / Tests

```bash
pip install pytest
pytest
```

Die Tests prüfen u. a. den 1-kHz-Referenzton, das 1-kHz-RTA-Band und die plattformneutralen Teile des Systemaudio-Backends.
