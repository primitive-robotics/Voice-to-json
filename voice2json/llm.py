"""
llm.py – LLM adapters for converting transcripts → robot JSON.

Configured via environment variables:
  LLM_PROVIDER  = anthropic | openai          (default: anthropic)
  ANTHROPIC_API_KEY                            (required for anthropic)
  OPENAI_API_KEY                               (required for openai)
  LLM_MODEL     = model name override          (optional)
"""

import json
import os
import textwrap
from typing import Optional

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Default model per provider
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a robot command parser. Convert the user's spoken instruction into a
    STRICT JSON object that conforms exactly to the robot_command_v0 schema below.

    RULES:
    1. Output ONLY valid JSON — no markdown fences, no commentary, no extra text.
    2. Use the current ISO 8601 local timestamp for the "timestamp" field.
    3. "type" must always be "robot_command_v0".
    4. "intent" must be one of: pick, place, move, inspect, pause, resume, stop, unknown.
    5. If the user says "stop", "abort", or "emergency stop": set intent="stop" and stop=true.
    6. If the command is ambiguous: set requires_confirmation=true and fill clarifying_question.
    7. If the text is NOT a robot instruction: set intent="unknown", requires_confirmation=true,
       clarifying_question="What do you want the robot to do?"
    8. "confidence" is a float 0.0–1.0 reflecting your certainty about the interpretation.
    9. Only include "target_description", "destination_description", "clarifying_question"
       when they add meaning (they are optional in the schema).

    JSON Schema:
    {
      "type": "object",
      "additionalProperties": false,
      "required": ["type", "intent", "requires_confirmation", "stop", "timestamp"],
      "properties": {
        "type": {"const": "robot_command_v0"},
        "intent": {"type": "string", "enum": ["pick","place","move","inspect","pause","resume","stop","unknown"]},
        "target_description": {"type": "string"},
        "destination_description": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "stop": {"type": "boolean"},
        "timestamp": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "clarifying_question": {"type": "string"}
      }
    }

    Examples:
    Input: "pick up the red box"
    Output: {"type":"robot_command_v0","intent":"pick","target_description":"red box","requires_confirmation":false,"stop":false,"timestamp":"<ISO8601>","confidence":0.95}

    Input: "stop"
    Output: {"type":"robot_command_v0","intent":"stop","requires_confirmation":false,"stop":true,"timestamp":"<ISO8601>","confidence":1.0}

    Input: "hello how are you"
    Output: {"type":"robot_command_v0","intent":"unknown","requires_confirmation":true,"stop":false,"timestamp":"<ISO8601>","confidence":0.1,"clarifying_question":"What do you want the robot to do?"}
""").strip()


def _build_user_message(
    transcript: str,
    error_feedback: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    parts = []
    if context:
        parts.append(context)
    parts.append(f"Transcript: {transcript}")
    if error_feedback:
        parts.append(
            f"\nYour previous JSON was INVALID. Error: {error_feedback}\n"
            "Fix the JSON and output ONLY valid JSON, nothing else."
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

def _call_anthropic(transcript: str, error_feedback: Optional[str] = None, context: Optional[str] = None) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it or add it to your .env file."
        )

    model = os.getenv("LLM_MODEL", _DEFAULT_MODELS["anthropic"])
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=LLM_TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _build_user_message(transcript, error_feedback, context)}
        ],
    )
    return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# OpenAI adapter (stub — same interface, swap in real call as needed)
# ---------------------------------------------------------------------------

def _call_openai(transcript: str, error_feedback: Optional[str] = None, context: Optional[str] = None) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Export it or add it to your .env file."
        )

    model = os.getenv("LLM_MODEL", _DEFAULT_MODELS["openai"])
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=LLM_TEMPERATURE,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(transcript, error_feedback, context)},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_command(
    transcript: str,
    error_feedback: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    """
    Call the configured LLM and return the raw response string.

    Args:
        transcript:     Whisper transcript text.
        error_feedback: Validation error from a previous attempt (for retries).
        context:        Optional prior-exchange context (for clarification rounds).

    Returns:
        Raw LLM output (should be JSON, but may need validation).
    """
    print(f"[llm] Using provider: {LLM_PROVIDER}", flush=True)
    if LLM_PROVIDER == "anthropic":
        return _call_anthropic(transcript, error_feedback, context)
    elif LLM_PROVIDER == "openai":
        return _call_openai(transcript, error_feedback, context)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. "
            "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai."
        )
