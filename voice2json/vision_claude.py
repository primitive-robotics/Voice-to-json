"""
vision_claude.py – Claude vision adapter for robot_vision_v0.

Encodes a camera frame as base64 and calls Claude's Messages API with
an image content block. Retries up to MAX_RETRIES times on schema failure.
"""

import base64
import os
import textwrap
from pathlib import Path
from typing import Optional, Tuple

from voice2json.schema import validate_with_retry, validate_vision_result

VISION_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
VISION_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
CONFIDENCE_THRESHOLD = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.6"))

VISION_SYSTEM_PROMPT = textwrap.dedent("""
    You are a precision robot vision system. Your job is to locate a target object
    in a camera image and return its exact pixel location.

    OUTPUT RULES:
    1. Output ONLY valid JSON — no markdown fences, no commentary, no extra text.
    2. The JSON must conform exactly to the robot_vision_v0 schema below.
    3. "type" must always be "robot_vision_v0".
    4. "timestamp" must be the current ISO 8601 local time.
    5. All pixel coordinates must be integers, clamped to image bounds.
    6. "x_pixel" and "y_pixel" are the bounding box center: x=(x1+x2)//2, y=(y1+y2)//2.
    7. "confidence" is your certainty (0.0–1.0) that you found the correct object.
    8. If confidence < 0.6: set requires_confirmation=true.
    9. If the object is not visible or unclear: found=false, requires_confirmation=true,
       clarifying_question="<specific question to help disambiguate>".
    10. If multiple matching candidates exist: found=true, requires_confirmation=true,
        clarifying_question="Did you mean the left one, the right one, the top or the bottom?"
    11. "label" describes what you found (e.g. "red cardboard box, approx 20×15 cm").
    12. Only include x_pixel, y_pixel, bbox, confidence, label when found=true.

    JSON Schema (robot_vision_v0):
    {
      "type": "object",
      "additionalProperties": false,
      "required": ["type", "found", "requires_confirmation", "timestamp"],
      "properties": {
        "type": {"const": "robot_vision_v0"},
        "found": {"type": "boolean"},
        "x_pixel": {"type": "integer"},
        "y_pixel": {"type": "integer"},
        "bbox": {
          "type": "object",
          "additionalProperties": false,
          "required": ["x1", "y1", "x2", "y2"],
          "properties": {
            "x1": {"type": "integer"}, "y1": {"type": "integer"},
            "x2": {"type": "integer"}, "y2": {"type": "integer"}
          }
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "clarifying_question": {"type": "string"},
        "timestamp": {"type": "string"}
      }
    }

    Examples:
    Found:
    {"type":"robot_vision_v0","found":true,"x_pixel":320,"y_pixel":240,"bbox":{"x1":200,"y1":150,"x2":440,"y2":330},"confidence":0.91,"label":"red cardboard box","requires_confirmation":false,"timestamp":"<ISO8601>"}

    Not found:
    {"type":"robot_vision_v0","found":false,"requires_confirmation":true,"clarifying_question":"I cannot see a red box in the frame. Is it out of the camera's field of view?","timestamp":"<ISO8601>"}
""").strip()


def _encode_image(frame_path: Path) -> str:
    """Return base64-encoded PNG bytes as a string."""
    with open(frame_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _call_claude_vision(
    frame_path: Path,
    target: str,
    image_size: Tuple[int, int],
    error_feedback: Optional[str] = None,
) -> str:
    """Single Claude vision API call. Returns raw text response."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    w, h = image_size
    b64 = _encode_image(frame_path)

    user_parts = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        },
        {
            "type": "text",
            "text": (
                f'Target object: "{target}"\n'
                f"Image dimensions: {w}×{h} pixels\n\n"
                "Locate the target object in the image above and return the "
                "robot_vision_v0 JSON. Output ONLY valid JSON."
            ),
        },
    ]

    if error_feedback:
        user_parts.append({
            "type": "text",
            "text": (
                f"\nYour previous JSON was INVALID. Error: {error_feedback}\n"
                "Fix it and output ONLY valid JSON, nothing else."
            ),
        })

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=VISION_MODEL,
        max_tokens=512,
        temperature=VISION_TEMPERATURE,
        system=VISION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_parts}],
    )
    return message.content[0].text.strip()


def run_vision_with_retry(
    frame_path: Path,
    target: str,
    image_size: Tuple[int, int],
    *,
    max_retries: int = 3,
) -> Tuple[dict, str, list[str]]:
    """
    Call Claude vision with retry until valid robot_vision_v0 JSON is returned.

    Args:
        frame_path:  Path to the captured PNG frame.
        target:      Target object description from the command.
        image_size:  (width, height) in pixels for the prompt.
        max_retries: Maximum attempts.

    Returns:
        (result_dict, raw_response, validation_errors)
    """
    print(f"[vision] Targeting: {target!r} in {frame_path.name}", flush=True)
    print(f"[vision] Model: {VISION_MODEL}", flush=True)

    def _llm_fn(_transcript: str, error_feedback: Optional[str] = None) -> str:
        return _call_claude_vision(frame_path, target, image_size, error_feedback)

    return validate_with_retry(
        "",
        _llm_fn,
        validator=validate_vision_result,
        max_retries=max_retries,
    )
