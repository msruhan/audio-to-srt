# Audio → SRT

Local tool to convert audio (`.wav` / `.m4a` / `.mp3` / …) into timed `.srt` subtitles using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Includes a Streamlit web UI and a PowerShell/CLI helper. Runs on your machine (CUDA GPU if available, otherwise CPU).

## Requirements

- Python 3.12+
- [ffmpeg](https://ffmpeg.org/) on `PATH` (required for `.m4a` and most non-WAV formats)
- NVIDIA GPU optional (faster); CPU works without it
- Internet once, to download the Whisper model on first run

## Setup (Windows)

```powershell
cd audio-to-srt
.\setup_local.ps1
```

## Web UI

```powershell
.\run_web.ps1
```

Open http://localhost:8501 — upload audio, transcribe, download `.srt`.

## CLI

```powershell
.\transcribe.ps1 -Engine local -Audio "C:\path\to\file.m4a" -Language id -Vad -Txt
```

Useful flags:

- `-Model small|medium|large-v3` (default in script helper: `large-v3`)
- `-Language id` / `en` / …
- `-Vad` — skip silence
- `-Txt` — also write plain transcript

Or call Python directly:

```powershell
.\.venv\Scripts\python.exe transcribe_local.py file.m4a --model small --language id --vad --txt
```

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit web UI |
| `transcribe_local.py` | Shared transcription + SRT writer |
| `transcribe.ps1` | CLI wrapper (`local` / optional `ot` cloud) |
| `setup_local.ps1` | Create venv + install deps |
| `run_web.ps1` | Start Streamlit |
| `requirements.txt` | Python dependencies |

`.venv` is not committed; always run `setup_local.ps1` on a new machine.
