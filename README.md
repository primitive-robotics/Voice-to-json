# voice2json — Phase 1 Prototype

> microphone → Whisper → LLM → robot-safe JSON

## Quick start

### 1. Prerequisites

- Python ≥ 3.10 (tested on conda base with 3.13)
- A working microphone
- An Anthropic or OpenAI API key

### 2. Install

```bash
# (recommanded) create/activate a conda env
conda create -n voice2json python=3.11 -y
conda activate voice2json

pip install -r requirements.txt
```

**Whisper note:** `requirements.txt` installs `faster-whisper` by default (fast
CPU inference with int8 quantization). If that fails, comment it out and
uncomment the `openai-whisper` line instead.

**PortAudio note:** `sounddevice` requires PortAudio. On Ubuntu/Debian:
```bash
sudo apt-get install portaudio19-dev
```
On macOS: `brew install portaudio`

### 3. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

### 4. Run

```bash
python -m voice2json
```

The tool will:
1. Print a banner with current settings
2. Wait for you to press **Enter** to start recording
3. Wait for you to press **Enter** again to stop recording (auto-stops after 30s)
4. Transcribe with Whisper
5. Convert to robot JSON via LLM (retries up to 3× on validation failure)
6. Print the validated JSON
7. Loop — press **Enter** to record again, or **Ctrl+C** to quit
8. If the command has `intent=stop`, the loop exits automatically

---

## Project structure

```
voice2json/
├── __init__.py       – package version
├── __main__.py       – entry point, .env loader
├── audio.py          – push-to-talk recording (sounddevice)
├── asr.py            – Whisper transcription (faster-whisper or openai-whisper)
├── llm.py            – Anthropic / OpenAI adapters + system prompt
├── app.py            – main loop + JSONL logging
└── schema.py         – JSON Schema + validator + retry logic

tests/
└── test_schema.py    – 27 unit tests (schema, retry, stop-command)

runs/
└── <timestamp>/
    ├── audio.wav
    └── log.jsonl     – one JSON line per event (success / error)
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic |
| `OPENAI_API_KEY` | — | Required for OpenAI |
| `LLM_MODEL` | provider default | Override model name |
| `LLM_TEMPERATURE` | `0.1` | 0–1, low = deterministic |
| `WHISPER_MODEL` | `base` | `tiny/base/small/medium/large-v2/large-v3` |
| `RUNS_DIR` | `./runs` | Where to save logs |

---

## JSON schema (robot_command_v0)

```json
{
  "type": "robot_command_v0",
  "intent": "pick | place | move | inspect | pause | resume | stop | unknown",
  "target_description": "optional string",
  "destination_description": "optional string",
  "requires_confirmation": true,
  "stop": false,
  "timestamp": "2025-01-01T12:00:00.000000",
  "confidence": 0.95,
  "clarifying_question": "optional string"
}
```

**Behavior rules (enforced in LLM prompt):**
- "stop / abort / emergency stop" → `intent=stop`, `stop=true`
- Ambiguous command → `requires_confirmation=true` + `clarifying_question`
- Non-robot text → `intent=unknown`, `requires_confirmation=true`
- Output is ONLY JSON — no markdown, no commentary

---

## Run tests

```bash
python -m pytest tests/ -v
```

All 27 tests pass on Python 3.10–3.13.



**Currently supported operations**
  ┌─────────┬──────────────────────────────────────────────────┐
  │ Intent  │                 Example phrases                  │
  ├─────────┼──────────────────────────────────────────────────┤
  │ pick    │ "pick up the red box", "grab the bolt"           │
  ├─────────┼──────────────────────────────────────────────────┤
  │ place   │ "put it on the shelf", "place the part in bin A" │
  ├─────────┼──────────────────────────────────────────────────┤
  │ move    │ "move to the left", "go forward 1 meter"         │
  ├─────────┼──────────────────────────────────────────────────┤
  │ inspect │ "inspect the weld", "check the part quality"     │
  ├─────────┼──────────────────────────────────────────────────┤
  │ pause   │ "pause", "hold on"                               │
  ├─────────┼──────────────────────────────────────────────────┤
  │ resume  │ "resume", "continue", "go ahead"                 │
  ├─────────┼──────────────────────────────────────────────────┤
  │ stop    │ "stop", "abort", "emergency stop"                │
  └─────────┴──────────────────────────────────────────────────┘


## TODO (Mar 2 2026):
Sequential execution: voice module only supports single commands and will not execute more than one command. Need to implement sequential execution

**Known bugs**:
N/a