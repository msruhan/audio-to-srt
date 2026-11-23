#!/usr/bin/env python3
"""Local audio → SRT (and optional TXT) via faster-whisper.

Supports .wav / .m4a / .mp3 / .flac / .ogg / .webm / .mp4 (ffmpeg required for non-wav).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SUPPORTED_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".webm", ".mp4", ".aac", ".wma"}


def prepend_nvidia_dll_dirs() -> None:
    """Make pip-installed NVIDIA CUDA wheels visible to ctranslate2 on Windows."""
    if sys.platform != "win32":
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    if not site_packages.is_dir():
        return
    added: list[str] = []
    for bin_dir in site_packages.glob("nvidia/*/bin"):
        path = str(bin_dir.resolve())
        if path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass
        added.append(path)
    if added:
        print(f"CUDA DLL dirs: {len(added)} nvidia wheel bin path(s)")


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        text = (seg.text or "").strip()
        if not text:
            continue
        start = format_srt_timestamp(float(seg.start))
        end = format_srt_timestamp(float(seg.end))
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def segments_to_txt(segments) -> str:
    text = "\n".join(
        (s.text or "").strip() for s in segments if (s.text or "").strip()
    )
    return text + ("\n" if text else "")


def write_srt(segments, out_path: Path) -> None:
    out_path.write_text(segments_to_srt(segments), encoding="utf-8")


def transcribe_file(
    audio: Path,
    *,
    model_name: str = "large-v3",
    language: str | None = None,
    device: str = "auto",
    vad: bool = False,
    beam_size: int = 5,
) -> tuple[list, dict]:
    """Transcribe a local audio file. Returns (segments, info_dict)."""
    prepend_nvidia_dll_dirs()
    audio = audio.resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"File not found: {audio}")
    if audio.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported extension {audio.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    from faster_whisper import WhisperModel

    device_name, compute_type = pick_device(device)

    with tempfile.TemporaryDirectory(prefix="audio-to-srt-") as tmp:
        wav = normalize_audio(audio, Path(tmp))
        model = WhisperModel(
            model_name,
            device=device_name,
            compute_type=compute_type,
        )

        def run_transcribe(active_model):
            segments_iter, info = active_model.transcribe(
                str(wav),
                language=language,
                beam_size=beam_size,
                vad_filter=vad,
                word_timestamps=True,
            )
            return list(segments_iter), {
                "language": info.language,
                "language_probability": float(info.language_probability),
                "device": device_name,
                "compute_type": compute_type,
                "model": model_name,
            }

        try:
            segments, meta = run_transcribe(model)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if device_name == "cuda" and (
                "cublas" in msg or "cuda" in msg or "cudnn" in msg
            ):
                model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                )
                segments, meta = run_transcribe(model)
                meta["device"] = "cpu"
                meta["compute_type"] = "int8"
                meta["fallback"] = str(exc)
            else:
                raise

    return segments, meta


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg, then retry "
            "(needed for .m4a and most non-wav formats)."
        )


def normalize_audio(src: Path, work_dir: Path) -> Path:
    """Convert to mono 16 kHz WAV for reliable Whisper input."""
    ensure_ffmpeg()
    dst = work_dir / "audio_16k_mono.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")
    return dst


def pick_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type)."""
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "float16"
    # auto
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe local audio to SRT with faster-whisper."
    )
    p.add_argument("audio", type=Path, help="Path to .wav / .m4a / etc.")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output .srt path (default: same name as input).",
    )
    p.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size/name (default: large-v3). "
        "Try medium or small for speed.",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. id / en. Omit for auto-detect.",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (default: auto).",
    )
    p.add_argument(
        "--vad",
        action="store_true",
        help="Enable VAD filter (helps skip long silence).",
    )
    p.add_argument(
        "--txt",
        action="store_true",
        help="Also write a plain .txt transcript next to the .srt.",
    )
    p.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size (default: 5).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    audio = args.audio.resolve()
    out_srt = args.out.resolve() if args.out else audio.with_suffix(".srt")
    out_srt.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Model: {args.model}")
        print(f"Input: {audio}")
        segments, meta = transcribe_file(
            audio,
            model_name=args.model,
            language=args.language,
            device=args.device,
            vad=args.vad,
            beam_size=args.beam_size,
        )
    except ImportError:
        raise SystemExit(
            "faster-whisper not installed. From tools/audio-to-srt run:\n"
            "  .\\setup_local.ps1"
        )
    except (FileNotFoundError, ValueError, SystemExit) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Detected language: {meta['language']} "
        f"(p={meta['language_probability']:.2f}) "
        f"on {meta['device']}/{meta['compute_type']}"
    )
    write_srt(segments, out_srt)
    print(f"Wrote SRT: {out_srt} ({len(segments)} cues)")

    if args.txt:
        out_txt = out_srt.with_suffix(".txt")
        out_txt.write_text(segments_to_txt(segments), encoding="utf-8")
        print(f"Wrote TXT: {out_txt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
