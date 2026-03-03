"""
Unit tests for schema.py:
  - validate_command() accepts / rejects various payloads
  - validate_with_retry() retries on bad output and succeeds when LLM fixes it
  - Stop-command behavior: intent=stop + stop=true
  - Unknown / ambiguous command behavior
"""

import json
import pytest
import datetime

from voice2json.schema import (
    validate_command,
    validate_with_retry,
    parse_json,
    ROBOT_COMMAND_SCHEMA,
    SEQUENCE_ITEM_SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.datetime.now().isoformat()


def _base(**kwargs) -> dict:
    """Return a minimal valid command, optionally overriding fields."""
    cmd = {
        "type": "robot_command_v0",
        "intent": "move",
        "requires_confirmation": False,
        "stop": False,
        "timestamp": _ts(),
    }
    cmd.update(kwargs)
    return cmd


# ---------------------------------------------------------------------------
# validate_command – positive cases
# ---------------------------------------------------------------------------

class TestValidateCommandValid:
    def test_minimal_valid(self):
        assert validate_command(_base()) is None

    def test_pick_with_target(self):
        cmd = _base(intent="pick", target_description="red box", confidence=0.9)
        assert validate_command(cmd) is None

    def test_stop_command(self):
        cmd = _base(intent="stop", stop=True, confidence=1.0)
        assert validate_command(cmd) is None

    def test_unknown_with_question(self):
        cmd = _base(
            intent="unknown",
            requires_confirmation=True,
            clarifying_question="What do you want the robot to do?",
        )
        assert validate_command(cmd) is None

    def test_all_intents_accepted(self):
        for intent in ["pick", "place", "move", "inspect", "pause", "resume", "stop", "unknown"]:
            stop = intent == "stop"
            assert validate_command(_base(intent=intent, stop=stop)) is None

    def test_confidence_boundaries(self):
        assert validate_command(_base(confidence=0.0)) is None
        assert validate_command(_base(confidence=1.0)) is None


# ---------------------------------------------------------------------------
# validate_command – negative cases
# ---------------------------------------------------------------------------

class TestValidateCommandInvalid:
    def test_missing_required_field(self):
        cmd = _base()
        del cmd["stop"]
        err = validate_command(cmd)
        assert err is not None
        assert "stop" in err

    def test_wrong_type_field(self):
        cmd = _base(type="wrong_type")
        err = validate_command(cmd)
        assert err is not None

    def test_invalid_intent(self):
        cmd = _base(intent="fly")
        err = validate_command(cmd)
        assert err is not None

    def test_confidence_out_of_range(self):
        cmd = _base(confidence=1.5)
        err = validate_command(cmd)
        assert err is not None

    def test_additional_property_rejected(self):
        cmd = _base(robot_arm="left")
        err = validate_command(cmd)
        assert err is not None

    def test_requires_confirmation_must_be_bool(self):
        cmd = _base(requires_confirmation="yes")
        err = validate_command(cmd)
        assert err is not None


# ---------------------------------------------------------------------------
# Stop-command semantic rules
# ---------------------------------------------------------------------------

class TestStopCommandRules:
    """The intent must be 'stop' AND stop must be True for emergency stops."""

    def test_stop_intent_with_stop_true(self):
        cmd = _base(intent="stop", stop=True)
        assert validate_command(cmd) is None

    def test_stop_intent_stop_false_still_schema_valid(self):
        # Schema doesn't enforce the semantic rule — that's the LLM's job.
        # We confirm schema still accepts it (no schema error).
        cmd = _base(intent="stop", stop=False)
        assert validate_command(cmd) is None

    def test_stop_false_with_move(self):
        cmd = _base(intent="move", stop=False)
        assert validate_command(cmd) is None


# ---------------------------------------------------------------------------
# parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_valid_json(self):
        raw = json.dumps(_base())
        parsed, err = parse_json(raw)
        assert err is None
        assert parsed["type"] == "robot_command_v0"

    def test_strips_markdown_fence(self):
        raw = "```json\n" + json.dumps(_base()) + "\n```"
        parsed, err = parse_json(raw)
        assert err is None

    def test_strips_plain_fence(self):
        raw = "```\n" + json.dumps(_base()) + "\n```"
        parsed, err = parse_json(raw)
        assert err is None

    def test_invalid_json_returns_error(self):
        parsed, err = parse_json("not json at all")
        assert parsed is None
        assert err is not None

    def test_empty_string_returns_error(self):
        parsed, err = parse_json("")
        assert parsed is None
        assert err is not None


# ---------------------------------------------------------------------------
# validate_with_retry
# ---------------------------------------------------------------------------

class TestValidateWithRetry:
    def test_success_on_first_attempt(self):
        valid_output = json.dumps(_base(intent="pick", target_description="blue cube"))

        def llm_fn(transcript, error_feedback=None):
            return valid_output

        cmd, raw, errors = validate_with_retry("pick up the blue cube", llm_fn)
        assert cmd["intent"] == "pick"
        assert errors == []

    def test_retries_on_bad_json_then_succeeds(self):
        call_count = [0]

        def llm_fn(transcript, error_feedback=None):
            call_count[0] += 1
            if call_count[0] < 2:
                return "not valid json"
            return json.dumps(_base(intent="move"))

        cmd, raw, errors = validate_with_retry("move forward", llm_fn)
        assert cmd["intent"] == "move"
        assert len(errors) == 1           # one failed attempt
        assert call_count[0] == 2

    def test_retries_on_schema_error_then_succeeds(self):
        call_count = [0]

        def llm_fn(transcript, error_feedback=None):
            call_count[0] += 1
            if call_count[0] < 3:
                # Missing required field 'stop'
                bad = _base()
                del bad["stop"]
                return json.dumps(bad)
            return json.dumps(_base(intent="inspect"))

        cmd, raw, errors = validate_with_retry("inspect the part", llm_fn)
        assert cmd["intent"] == "inspect"
        assert len(errors) == 2
        assert call_count[0] == 3

    def test_exhausts_retries_raises(self):
        def llm_fn(transcript, error_feedback=None):
            return "always broken"

        with pytest.raises(ValueError, match="failed to produce valid JSON"):
            validate_with_retry("anything", llm_fn, max_retries=3)

    def test_error_feedback_passed_on_retry(self):
        feedbacks = []

        def llm_fn(transcript, error_feedback=None):
            feedbacks.append(error_feedback)
            if error_feedback is None:
                return "bad json"
            return json.dumps(_base())

        validate_with_retry("move left", llm_fn)
        assert feedbacks[0] is None          # first call: no feedback
        assert feedbacks[1] is not None      # second call: error feedback provided


# ---------------------------------------------------------------------------
# Sequence commands
# ---------------------------------------------------------------------------

class TestSequenceCommands:
    def _step(self, **kwargs) -> dict:
        step = {"intent": "move", "requires_confirmation": False, "stop": False}
        step.update(kwargs)
        return step

    def test_valid_sequence(self):
        cmd = _base(
            intent="sequence",
            sequence=[
                self._step(intent="move", target_description="position 3"),
                self._step(intent="pick", target_description="red box"),
            ],
        )
        assert validate_command(cmd) is None

    def test_sequence_requires_two_or_more_steps(self):
        cmd = _base(
            intent="sequence",
            sequence=[self._step(intent="move")],  # only 1 — violates minItems:2
        )
        err = validate_command(cmd)
        assert err is not None

    def test_sequence_step_invalid_intent(self):
        cmd = _base(
            intent="sequence",
            sequence=[
                self._step(intent="fly"),           # invalid intent
                self._step(intent="pick"),
            ],
        )
        err = validate_command(cmd)
        assert err is not None

    def test_sequence_step_additional_property_rejected(self):
        cmd = _base(
            intent="sequence",
            sequence=[
                {**self._step(intent="move"), "robot_arm": "left"},  # extra field
                self._step(intent="pick"),
            ],
        )
        err = validate_command(cmd)
        assert err is not None

    def test_sequence_with_stop_step(self):
        cmd = _base(
            intent="sequence",
            sequence=[
                self._step(intent="move", target_description="position 3"),
                self._step(intent="stop", stop=True),
            ],
        )
        assert validate_command(cmd) is None

    def test_sequence_intent_accepted_in_schema(self):
        # "sequence" is now a valid top-level intent
        cmd = _base(intent="sequence", sequence=[
            self._step(intent="pick"),
            self._step(intent="place"),
        ])
        assert validate_command(cmd) is None


# ---------------------------------------------------------------------------
# Ambiguous / unknown command rules
# ---------------------------------------------------------------------------

class TestAmbiguousCommands:
    def test_unknown_intent_requires_confirmation(self):
        cmd = _base(
            intent="unknown",
            requires_confirmation=True,
            clarifying_question="What do you want the robot to do?",
        )
        assert validate_command(cmd) is None

    def test_ambiguous_pick_no_target(self):
        # pick without target_description is schema-valid but ambiguous
        cmd = _base(intent="pick", requires_confirmation=True,
                    clarifying_question="Which object should I pick?")
        assert validate_command(cmd) is None
