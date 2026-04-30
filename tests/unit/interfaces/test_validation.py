"""Unit tests for the validation interface."""

from llm_evolution.interfaces.validation import (
    AlwaysValidValidator,
    InstanceValidator,
    ValidationResult,
)


def test_always_valid_validator_satisfies_protocol():
    v = AlwaysValidValidator()
    assert isinstance(v, InstanceValidator)


def test_always_valid_validator_returns_ok():
    v = AlwaysValidValidator()
    result = v("anything")
    assert result.ok is True
    assert result.as_feedback() == ""


def test_validation_result_as_feedback_with_stage():
    r = ValidationResult(ok=False, stage="compile", feedback="missing semicolon")
    assert r.as_feedback() == "[compile] missing semicolon"


def test_validation_result_as_feedback_without_stage():
    r = ValidationResult(ok=False, feedback="bad output")
    assert r.as_feedback() == "bad output"


def test_validation_result_ok_returns_empty_feedback():
    r = ValidationResult(ok=True, feedback="ignored")
    assert r.as_feedback() == ""
