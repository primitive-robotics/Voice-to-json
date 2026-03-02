"""
audio.py – Push-to-talk audio recording.

Interaction:
  Press Enter  → start recording
  Press Enter  → stop recording  (or it auto-stops after MAX_SECONDS)

Returns a WAV file path.
"""

import os
import sys
import threading
import wave
import time
from pathlib import Path
from typing import Optional

SAMPLE_RATE = 16_000   # Whisper prefers 16 kHz
CHANNELS = 1
MAX_SECONDS = 30       # safety cap
DTYPE = "int16"


def _wait_for_enter(event: threading.Event) -> None:
    """Block until the user presses Enter, then set the event."""
    input()
    event.set()


def record_audio(output_path: Path, *, verbose: bool = True) -> Path:
    """
    Record from the default microphone using push-to-talk.

    Raises:
        RuntimeError: if sounddevice or a microphone is unavailable.
    Returns:
        Path to the saved WAV file.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        raise RuntimeError(
            "sounddevice is not installed. Run: pip install sounddevice"
        )

    # Verify a mic exists
    try:
        sd.check_input_settings(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    except Exception as exc:
        raise RuntimeError(
            f"Microphone check failed: {exc}\n"
            "Make sure a microphone is connected and accessible."
        ) from exc

    if verbose:
        print("\n[audio] Press ENTER to start recording…", flush=True)

    stop_event = threading.Event()
    t = threading.Thread(target=_wait_for_enter, args=(stop_event,), daemon=True)
    t.start()
    stop_event.wait()          # wait for first Enter

    if verbose:
        print(f"[audio] Recording… (press ENTER to stop, max {MAX_SECONDS}s)", flush=True)

    frames: list = []
    stop_event.clear()

    # Start a fresh listener for the second Enter
    t2 = threading.Thread(target=_wait_for_enter, args=(stop_event,), daemon=True)
    t2.start()

    start = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as stream:
        while not stop_event.is_set():
            if time.time() - start > MAX_SECONDS:
                if verbose:
                    print(f"\n[audio] Max {MAX_SECONDS}s reached — stopping automatically.")
                break
            chunk, _ = stream.read(1024)
            frames.append(chunk)

    if verbose:
        elapsed = time.time() - start
        print(f"[audio] Recorded {elapsed:.1f}s of audio.", flush=True)

    # Save WAV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    audio_data = np.concatenate(frames, axis=0)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    if verbose:
        print(f"[audio] Saved → {output_path}", flush=True)

    return output_path
