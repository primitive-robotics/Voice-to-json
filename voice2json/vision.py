"""
vision.py – Phase 2: webcam capture, overlay rendering, and vision orchestration.

VISION_CAPTURE_MODE=manual  → show live feed; press SPACE to capture, ESC to quit
VISION_CAPTURE_MODE=auto    → capture one frame immediately (headless-friendly)
VISION_CAMERA_ID=0          → OpenCV camera device index (default 0)
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

CAPTURE_MODE = os.getenv("VISION_CAPTURE_MODE", "manual")
CAMERA_ID = int(os.getenv("VISION_CAMERA_ID", "0"))
CONFIDENCE_THRESHOLD = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.6"))

# Intents that are object-targeting (need vision); others don't benefit from it
_TARGETING_INTENTS = {"pick", "place", "inspect"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _skip_result(
    reason: str,
    *,
    requires_confirmation: bool,
    clarifying_question: Optional[str] = None,
) -> dict:
    """Return a robot_vision_v0 result indicating vision was skipped."""
    print(f"[vision] Skipping vision: {reason}", flush=True)
    result: dict = {
        "type": "robot_vision_v0",
        "found": False,
        "requires_confirmation": requires_confirmation,
        "timestamp": _now_iso(),
    }
    if clarifying_question:
        result["clarifying_question"] = clarifying_question
    return result


def _get_target(command: dict) -> Optional[str]:
    """Extract the most relevant target description from a command or sequence."""
    if command.get("intent") == "sequence":
        for step in command.get("sequence", []):
            if step.get("intent") in _TARGETING_INTENTS and step.get("target_description"):
                return step["target_description"]
        # Fallback: first step with any target
        for step in command.get("sequence", []):
            if step.get("target_description"):
                return step["target_description"]
        return None
    return command.get("target_description")


# ---------------------------------------------------------------------------
# Webcam capture
# ---------------------------------------------------------------------------

def capture_frame(output_path: Path, mode: str = CAPTURE_MODE) -> Path:
    """
    Capture a single frame from the webcam.

    Args:
        output_path: Where to save the PNG frame.
        mode:        "manual" (press SPACE) or "auto" (capture immediately).

    Returns:
        Path to the saved frame.

    Raises:
        RuntimeError: if the camera is unavailable or no frame is captured.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "opencv-python is not installed. Run: pip install opencv-python"
        )

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam (device {CAMERA_ID}). "
            "Check that a camera is connected and not in use by another application."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "auto":
        # Warm up: discard first few frames so auto-exposure can settle
        for _ in range(8):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to read a frame from the webcam.")
        cv2.imwrite(str(output_path), frame)
        print(f"[vision] Auto-captured frame → {output_path}", flush=True)
        return output_path

    # Manual mode: show live feed
    window_name = "voice2json — SPACE: capture  |  ESC: skip"
    captured_frame = None
    MANUAL_TIMEOUT = 30  # seconds before auto-capturing

    # Try to create a named window; if that fails immediately, go auto
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
    except cv2.error:
        cap.release()
        print("[vision] Cannot create display window; falling back to auto capture.", flush=True)
        return capture_frame(output_path, mode="auto")

    print(
        f"[vision] Camera feed open (timeout {MANUAL_TIMEOUT}s).\n"
        "         Press SPACE in the camera window to capture, ESC to skip.",
        flush=True,
    )

    start = time.time()
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            try:
                cv2.imshow(window_name, img)
                # If the window is not visible (Wayland / headless), WND_PROP_VISIBLE < 1
                visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = -1

            if visible < 1:
                print("[vision] Window not visible; falling back to auto capture.", flush=True)
                cap.release()
                cv2.destroyAllWindows()
                return capture_frame(output_path, mode="auto")

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                captured_frame = img.copy()
                break
            elif key == 27:  # ESC
                break

            if time.time() - start > MANUAL_TIMEOUT:
                print(f"[vision] {MANUAL_TIMEOUT}s timeout — auto-capturing.", flush=True)
                captured_frame = img.copy()
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured_frame is None:
        raise RuntimeError("No frame captured (ESC pressed or camera error).")

    cv2.imwrite(str(output_path), captured_frame)
    print(f"[vision] Frame captured → {output_path}", flush=True)
    return output_path


def get_frame_size(frame_path: Path) -> Tuple[int, int]:
    """Return (width, height) of a saved frame."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is not installed.")
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Cannot read frame at {frame_path}")
    h, w = img.shape[:2]
    return w, h


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def draw_overlay(frame_path: Path, result: dict, output_path: Path) -> Path:
    """
    Draw bounding box + crosshair on the frame and save as overlay.png.

    Only draws when result["found"] is True and "bbox" is present.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is not installed.")

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {frame_path}")

    if result.get("found") and result.get("bbox"):
        bbox = result["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        cx = result.get("x_pixel", (x1 + x2) // 2)
        cy = result.get("y_pixel", (y1 + y2) // 2)
        conf = result.get("confidence", 0.0)
        label = result.get("label", "target")

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crosshair at center
        cv2.drawMarker(
            frame, (cx, cy), (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2,
        )

        # Label + confidence
        text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    else:
        # Object not found — draw a red "NOT FOUND" banner
        h, w = frame.shape[:2]
        msg = result.get("clarifying_question", "Object not found")
        cv2.putText(
            frame, "NOT FOUND", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
        )
        cv2.putText(
            frame, msg[:80], (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
        )

    output_path = Path(output_path)
    cv2.imwrite(str(output_path), frame)
    print(f"[vision] Overlay saved → {output_path}", flush=True)
    return output_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_vision(command: dict, run_dir: Path, *, verbose: bool = True) -> dict:
    """
    Phase 2: capture frame → Claude vision → validate → log files.

    Bypass rules (per spec):
      - stop command         → found=false, requires_confirmation=false
      - requires_confirmation or unknown intent → found=false, requires_confirmation=true

    Returns:
        Validated robot_vision_v0 dict.
    """
    from voice2json.vision_claude import run_vision_with_retry

    # ── Bypass: stop ──────────────────────────────────────────────────────
    if command.get("stop") is True or command.get("intent") == "stop":
        return _skip_result(
            "stop command received",
            requires_confirmation=False,
        )

    # ── Bypass: unresolved ambiguity ─────────────────────────────────────
    if command.get("requires_confirmation") or command.get("intent") == "unknown":
        return _skip_result(
            "command requires clarification before targeting",
            requires_confirmation=True,
            clarifying_question=command.get(
                "clarifying_question",
                "Please clarify the command before I attempt visual targeting.",
            ),
        )

    # ── Extract target ────────────────────────────────────────────────────
    target = _get_target(command)
    if not target:
        return _skip_result(
            "no target_description in command",
            requires_confirmation=True,
            clarifying_question="Which object should I target?",
        )

    # ── Capture frame ─────────────────────────────────────────────────────
    frame_path = run_dir / "frame.png"
    try:
        capture_frame(frame_path, mode=CAPTURE_MODE)
    except RuntimeError as exc:
        print(f"[vision] Camera error: {exc}", file=sys.stderr)
        return _skip_result(
            str(exc),
            requires_confirmation=True,
            clarifying_question="Camera unavailable. Please check the connection.",
        )

    image_size = get_frame_size(frame_path)

    # ── Claude vision call ────────────────────────────────────────────────
    print("\n[vision] Calling Claude vision…", flush=True)
    try:
        result, raw, errors = run_vision_with_retry(frame_path, target, image_size)
    except ValueError as exc:
        print(f"[vision] Vision LLM exhausted retries: {exc}", file=sys.stderr)
        result = _skip_result(
            "LLM failed to return valid JSON",
            requires_confirmation=True,
            clarifying_question="Vision model could not parse the scene. Try again.",
        )
        raw = str(exc)
        errors = []

    # ── Post-process: clamp bbox + apply confidence threshold ─────────────
    if result.get("found") and result.get("bbox"):
        w, h = image_size
        b = result["bbox"]
        b["x1"] = max(0, min(b["x1"], w - 1))
        b["y1"] = max(0, min(b["y1"], h - 1))
        b["x2"] = max(0, min(b["x2"], w - 1))
        b["y2"] = max(0, min(b["y2"], h - 1))
        result["x_pixel"] = (b["x1"] + b["x2"]) // 2
        result["y_pixel"] = (b["y1"] + b["y2"]) // 2

    if result.get("found") and result.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
        result["requires_confirmation"] = True
        if not result.get("clarifying_question"):
            result["clarifying_question"] = (
                f"Low confidence ({result['confidence']:.0%}). "
                "Is this the correct object?"
            )

    # ── Draw overlay ──────────────────────────────────────────────────────
    try:
        draw_overlay(frame_path, result, run_dir / "overlay.png")
    except RuntimeError as exc:
        print(f"[vision] Overlay skipped: {exc}", flush=True)

    # ── Persist artifacts ─────────────────────────────────────────────────
    (run_dir / "raw_model_response.txt").write_text(raw, encoding="utf-8")
    (run_dir / "parsed_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    return result
