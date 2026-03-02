"""
asr.py – Automatic Speech Recognition via Whisper.

Tries faster-whisper first (much faster on CPU/GPU); falls back to openai-whisper.
Model size is configurable via WHISPER_MODEL env var (default: "base").
"""

import os
from pathlib import Path

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


def _transcribe_faster_whisper(audio_path: Path) -> str:
    from faster_whisper import WhisperModel  # type: ignore
    print(f"[asr] Loading faster-whisper model '{WHISPER_MODEL}'…", flush=True)
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"[asr] Detected language: {info.language} (prob={info.language_probability:.2f})")
    return text


def _transcribe_openai_whisper(audio_path: Path) -> str:
    import whisper  # type: ignore
    print(f"[asr] Loading openai-whisper model '{WHISPER_MODEL}'…", flush=True)
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(str(audio_path))
    return result["text"].strip()


def transcribe(audio_path: Path) -> str:
    """
    Transcribe *audio_path* to text.

    Tries faster-whisper → openai-whisper → raises RuntimeError.

    Returns:
        Transcribed text string.
    Raises:
        RuntimeError: if neither Whisper backend is installed.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    errors: list[str] = []

    try:
        text = _transcribe_faster_whisper(audio_path)
        print(f"[asr] (faster-whisper) Transcript: {text!r}", flush=True)
        return text
    except ImportError:
        errors.append("faster-whisper not installed")

    try:
        text = _transcribe_openai_whisper(audio_path)
        print(f"[asr] (openai-whisper) Transcript: {text!r}", flush=True)
        return text
    except ImportError:
        errors.append("openai-whisper (whisper) not installed")

    raise RuntimeError(
        "No Whisper backend found.\n"
        "Install one of:\n"
        "  pip install faster-whisper\n"
        "  pip install openai-whisper\n"
        f"Details: {'; '.join(errors)}"
    )
