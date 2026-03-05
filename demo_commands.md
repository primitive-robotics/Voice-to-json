# Demo Commands — voice2json Phase 1 + 2

Five example voice commands to exercise the full pipeline.
Each shows what you say, the Phase 1 JSON, and the expected Phase 2 vision behaviour.

---

## 1. Simple pick — object targeting

**Say:** `"pick up the red box"`

**Phase 1 output:**
```json
{
  "type": "robot_command_v0",
  "intent": "pick",
  "target_description": "red box",
  "requires_confirmation": false,
  "stop": false,
  "timestamp": "...",
  "confidence": 0.95
}
```

**Phase 2 behaviour:** Camera opens → point at a red box → press SPACE.
Claude locates it, returns bbox + x_pixel/y_pixel. `overlay.png` shows green rectangle + red crosshair.

---

## 2. Sequence — two steps

**Say:** `"move to position 3 and then pick up the red box"`

**Phase 1 output:**
```json
{
  "type": "robot_command_v0",
  "intent": "sequence",
  "requires_confirmation": false,
  "stop": false,
  "timestamp": "...",
  "sequence": [
    { "intent": "move", "target_description": "position 3", "requires_confirmation": false, "stop": false },
    { "intent": "pick", "target_description": "red box",    "requires_confirmation": false, "stop": false }
  ]
}
```

**Phase 2 behaviour:** Targets the first pick/inspect step with `target_description` — "red box". Move steps don't require visual targeting.

---

## 3. Inspect — quality check

**Say:** `"inspect the weld on the left bracket"`

**Phase 1 output:**
```json
{
  "type": "robot_command_v0",
  "intent": "inspect",
  "target_description": "weld on the left bracket",
  "requires_confirmation": false,
  "stop": false,
  "timestamp": "...",
  "confidence": 0.88
}
```

**Phase 2 behaviour:** Camera captures the work surface. Claude locates the weld seam and returns centroid coordinates. Low-confidence result triggers `requires_confirmation=true`.

---

## 4. Emergency stop

**Say:** `"stop"` (or `"abort"` / `"emergency stop"`)

**Phase 1 output:**
```json
{
  "type": "robot_command_v0",
  "intent": "stop",
  "requires_confirmation": false,
  "stop": true,
  "timestamp": "...",
  "confidence": 1.0
}
```

**Phase 2 behaviour:** Vision is **skipped** immediately. Returns:
```json
{ "type": "robot_vision_v0", "found": false, "requires_confirmation": false, "timestamp": "..." }
```
The main loop exits after printing the stop command.

---

## 5. Ambiguous command — clarification flow

**Say:** `"put it there"`

**Phase 1 output:**
```json
{
  "type": "robot_command_v0",
  "intent": "place",
  "requires_confirmation": true,
  "stop": false,
  "timestamp": "...",
  "confidence": 0.3,
  "clarifying_question": "What object should I place, and where should I put it?"
}
```

**Phase 2 behaviour:** Vision is **skipped** (command is ambiguous). Returns `found=false, requires_confirmation=true`. The app re-records your verbal answer, resolves the ambiguity, then runs a new full cycle with vision enabled.
