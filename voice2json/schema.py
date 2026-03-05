"""
schema.py – JSON Schema definition and validator with retry logic.

The schema enforces robot_command_v0 structure.
validate_with_retry() calls the LLM up to MAX_RETRIES times until the
output passes validation.
"""

import json
import re
from typing import Any, Callable, Optional, Tuple, Union

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    raise RuntimeError(
        "jsonschema is not installed. Run: pip install jsonschema"
    )

MAX_RETRIES = 3

_INTENTS = ["pick", "place", "move", "inspect", "pause", "resume", "stop", "unknown"]

# Schema for each step inside a sequence (no type/timestamp/sequence nesting)
SEQUENCE_ITEM_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "requires_confirmation", "stop"],
    "properties": {
        "intent": {"type": "string", "enum": _INTENTS},
        "target_description": {"type": "string"},
        "destination_description": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "stop": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "clarifying_question": {"type": "string"},
    },
}

ROBOT_COMMAND_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "intent", "requires_confirmation", "stop", "timestamp"],
    "properties": {
        "type": {"const": "robot_command_v0"},
        "intent": {
            "type": "string",
            "enum": _INTENTS + ["sequence"],
        },
        "target_description": {"type": "string"},
        "destination_description": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "stop": {"type": "boolean"},
        "timestamp": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "clarifying_question": {"type": "string"},
        "sequence": {
            "type": "array",
            "items": SEQUENCE_ITEM_SCHEMA,
            "minItems": 2,
        },
    },
}


ROBOT_VISION_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "found", "requires_confirmation", "timestamp"],
    "properties": {
        "type": {"const": "robot_vision_v0"},
        "found": {"type": "boolean"},
        "x_pixel": {"type": "integer"},
        "y_pixel": {"type": "integer"},
        "bbox": {
            "type": "object",
            "additionalProperties": False,
            "required": ["x1", "y1", "x2", "y2"],
            "properties": {
                "x1": {"type": "integer"},
                "y1": {"type": "integer"},
                "x2": {"type": "integer"},
                "y2": {"type": "integer"},
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "clarifying_question": {"type": "string"},
        "timestamp": {"type": "string"},
    },
}


def validate_vision_result(data: dict) -> Optional[str]:
    """Validate *data* against ROBOT_VISION_SCHEMA. Returns None if valid, else error string."""
    try:
        validate(instance=data, schema=ROBOT_VISION_SCHEMA)
        return None
    except ValidationError as exc:
        path = " → ".join(str(p) for p in exc.absolute_path) if exc.absolute_path else "root"
        return f"Schema validation error at '{path}': {exc.message}"


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` wrappers if the LLM added them."""
    text = text.strip()
    # Remove opening fence (```json or ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json(raw: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Attempt to parse *raw* as JSON.

    Returns:
        (parsed_dict, None)         on success
        (None, error_message)       on failure
    """
    cleaned = _strip_markdown_fences(raw)
    try:
        data = json.loads(cleaned)
        return data, None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def validate_command(data: dict) -> Optional[str]:
    """
    Validate *data* against ROBOT_COMMAND_SCHEMA.

    Returns:
        None              if valid
        error string      if invalid
    """
    try:
        validate(instance=data, schema=ROBOT_COMMAND_SCHEMA)
        return None
    except ValidationError as exc:
        # Return a concise path + message for the LLM retry prompt
        path = " → ".join(str(p) for p in exc.absolute_path) if exc.absolute_path else "root"
        return f"Schema validation error at '{path}': {exc.message}"


def validate_with_retry(
    transcript: str,
    llm_fn: Callable[[str, Optional[str]], str],
    *,
    max_retries: int = MAX_RETRIES,
    validator: Optional[Callable[[dict], Optional[str]]] = None,
) -> Tuple[dict, str, list[str]]:
    """
    Call *llm_fn* and validate the result, retrying on failure.

    Args:
        transcript:  Whisper transcript.
        llm_fn:      Callable(transcript, error_feedback) → raw LLM string.
        max_retries: Maximum number of attempts.

    Returns:
        (command_dict, raw_llm_output, validation_errors)
        where validation_errors is a list of error strings (empty = success on first try).

    Raises:
        ValueError: if all retries are exhausted without valid output.
    """
    errors: list[str] = []
    last_raw = ""
    error_feedback: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        print(f"[schema] LLM attempt {attempt}/{max_retries}…", flush=True)
        raw = llm_fn(transcript, error_feedback)
        last_raw = raw

        parsed, parse_err = parse_json(raw)
        if parse_err:
            error_feedback = parse_err
            errors.append(f"Attempt {attempt}: {parse_err}")
            print(f"[schema] {parse_err}", flush=True)
            continue

        _validator = validator if validator is not None else validate_command
        val_err = _validator(parsed)
        if val_err:
            error_feedback = val_err
            errors.append(f"Attempt {attempt}: {val_err}")
            print(f"[schema] {val_err}", flush=True)
            continue

        # Success
        print(f"[schema] Valid JSON on attempt {attempt}.", flush=True)
        return parsed, raw, errors

    raise ValueError(
        f"LLM failed to produce valid JSON after {max_retries} attempts.\n"
        f"Last output: {last_raw!r}\n"
        f"Errors: {errors}"
    )
