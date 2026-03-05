"""
app.py – Main application loop.

Orchestrates: audio recording → ASR → LLM → schema validation → JSONL logging.
"""

import json
import os
import sys
import datetime
from pathlib import Path
from typing import Optional

from voice2json.audio import record_audio
from voice2json.asr import transcribe
from voice2json.llm import generate_command
from voice2json.schema import validate_with_retry
from voice2json.vision import run_vision

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _run_dir() -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    d = RUNS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log(run_dir: Path, record: dict) -> None:
    log_file = run_dir / "log.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Single interaction
# ---------------------------------------------------------------------------

def run_once(run_dir: Path, *, verbose: bool = True) -> dict:
    """
    Perform one push-to-talk → ASR → LLM → validate cycle.

    Returns the validated command dict.
    """
    # 1. Record
    audio_path = run_dir / "audio.wav"
    try:
        record_audio(audio_path, verbose=verbose)
    except RuntimeError as exc:
        print(f"\n[error] Audio recording failed:\n  {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. ASR
    print("\n[asr] Transcribing…", flush=True)
    try:
        transcript = transcribe(audio_path)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\n[error] ASR failed:\n  {exc}", file=sys.stderr)
        _log(run_dir, {
            "event": "asr_error",
            "audio_file": str(audio_path),
            "error": str(exc),
        })
        sys.exit(1)

    print(f"\n[app] Transcript: {transcript!r}", flush=True)

    # 3. LLM + validation (with retry)
    print("\n[llm] Generating command…", flush=True)
    try:
        command, raw_output, val_errors = validate_with_retry(
            transcript,
            generate_command,
        )
    except ValueError as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        _log(run_dir, {
            "event": "llm_exhausted",
            "audio_file": str(audio_path),
            "transcript": transcript,
            "error": str(exc),
        })
        sys.exit(1)

    # 4. Clarification round — if LLM flagged ambiguity, record a follow-up
    if command.get("requires_confirmation") and command.get("clarifying_question"):
        question = command["clarifying_question"]
        print(f"\n[app] Clarification needed: {question}", flush=True)
        print("[app] Please record your answer:", flush=True)

        followup_path = run_dir / "audio_followup.wav"
        try:
            record_audio(followup_path, verbose=verbose)
            followup_transcript = transcribe(followup_path)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"\n[error] Follow-up recording failed: {exc}", file=sys.stderr)
            # Return the ambiguous command as-is rather than crashing
        else:
            print(f"\n[app] Follow-up: {followup_transcript!r}", flush=True)

            # Build context so the LLM understands the full exchange
            context = (
                f"Original command: '{transcript}'\n"
                f"Clarifying question asked: '{question}'\n"
                f"User's clarification: '{followup_transcript}'"
            )

            def _llm_with_context(t: str, err=None) -> str:
                return generate_command(t, error_feedback=err, context=context)

            try:
                command, raw_output, val_errors = validate_with_retry(
                    followup_transcript,
                    _llm_with_context,
                )
                _log(run_dir, {
                    "event": "clarification",
                    "followup_audio": str(followup_path),
                    "followup_transcript": followup_transcript,
                    "context": context,
                    "raw_llm_output": raw_output,
                    "validation_errors": val_errors,
                    "command": command,
                })
            except ValueError as exc:
                print(f"\n[error] {exc}", file=sys.stderr)

    # 5. Log
    _log(run_dir, {
        "event": "success",
        "audio_file": str(audio_path),
        "transcript": transcript,
        "raw_llm_output": raw_output,
        "validation_errors": val_errors,
        "command": command,
    })

    return command


# ---------------------------------------------------------------------------
# Continuous loop
# ---------------------------------------------------------------------------

def run_loop() -> None:
    """Run the push-to-talk loop until the user interrupts or says 'stop'."""
    _print_banner()

    while True:
        run_dir = _run_dir()
        print(f"\n{'─' * 60}", flush=True)
        print(f"[app] Run dir: {run_dir}", flush=True)

        try:
            command = run_once(run_dir)
        except SystemExit:
            break

        # ── Phase 2: Vision ───────────────────────────────────────────────
        print("\n[app] Phase 2 — Vision targeting…", flush=True)
        vision_result = run_vision(command, run_dir)
        print("\n[app] ✓ Vision result:")
        print(json.dumps(vision_result, indent=2))
        _log(run_dir, {"event": "vision", "vision_result": vision_result})

        # ── Pretty-print Phase 1 result and dispatch ──────────────────────
        if command.get("intent") == "sequence":
            steps = command.get("sequence", [])
            print(f"\n[app] ✓ Sequence of {len(steps)} commands:", flush=True)
            halted = False
            for i, step in enumerate(steps, 1):
                print(f"\n[app] Step {i}/{len(steps)}:")
                print(json.dumps(step, indent=2))
                if step.get("stop") is True or step.get("intent") == "stop":
                    print("\n[app] STOP step in sequence — halting.", flush=True)
                    halted = True
                    break
            if halted or command.get("stop") is True:
                break
        else:
            print("\n[app] ✓ Validated command:")
            print(json.dumps(command, indent=2))
            if command.get("stop") is True or command.get("intent") == "stop":
                print("\n[app] STOP command received — exiting loop.", flush=True)
                break

        # Ask to continue
        print("\n[app] Press ENTER to record another command, or Ctrl+C to quit.")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print("\n[app] Exiting.", flush=True)
            break


def _print_banner() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════╗\n"
        "║         voice2json  –  Phase 1 + 2           ║\n"
        "║  mic → Whisper → LLM → Vision → robot JSON  ║\n"
        "╚══════════════════════════════════════════════╝\n"
        "\n"
        "Controls:\n"
        "  Enter  →  start / stop recording\n"
        "  Ctrl+C →  quit at any time\n"
        "\n"
        f"Provider : {os.getenv('LLM_PROVIDER','anthropic')}\n"
        f"Model    : {os.getenv('LLM_MODEL','(default)')}\n"
        f"Vision   : CLAUDE_MODEL={os.getenv('CLAUDE_MODEL','claude-sonnet-4-6')}  "
        f"CAPTURE={os.getenv('VISION_CAPTURE_MODE','manual')}\n"
        f"ASR      : WHISPER_MODEL={os.getenv('WHISPER_MODEL','base')}\n"
        f"Runs dir : {RUNS_DIR.resolve()}\n",
        flush=True,
    )
