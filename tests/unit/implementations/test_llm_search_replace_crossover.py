"""Unit tests for LLMSearchReplaceCrossover."""

from __future__ import annotations

import random

import pytest

from llm_evolution.ai.interfaces.llm import Message
from llm_evolution.implementations.llm_search_replace_crossover import (
    LLMSearchReplaceCrossover,
)
from llm_evolution.interfaces.crossover import Crossover
from llm_evolution.interfaces.validation import ValidationResult


class _StubLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def __call__(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        return self._responses.pop(0)


class _RecordingBuilder:
    def __init__(self):
        self.last_args: tuple[str, str] | None = None

    def __call__(self, base: str, donor: str) -> list[Message]:
        self.last_args = (base, donor)
        return [
            Message(role="system", content="cross"),
            Message(role="user", content=f"BASE\n{base}\nDONOR\n{donor}"),
        ]


def _block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


def test_implements_crossover_protocol():
    op = LLMSearchReplaceCrossover(_StubLLM([]), _RecordingBuilder())
    assert isinstance(op, Crossover)


def test_returns_empty_when_fewer_than_two_parents():
    op = LLMSearchReplaceCrossover(_StubLLM([]), _RecordingBuilder())
    assert op([]) == []
    assert op(["only"]) == []


def test_returns_empty_when_probability_zero():
    op = LLMSearchReplaceCrossover(
        _StubLLM([]), _RecordingBuilder(), crossover_probability=0.0
    )
    assert op(["a", "b"]) == []


def test_offspring_returned_on_success():
    rng = random.Random(0)
    rng.random = lambda: 0.0  # type: ignore[method-assign]
    llm = _StubLLM([_block("aaa", "AAA")])
    op = LLMSearchReplaceCrossover(llm, _RecordingBuilder(), rng=rng)
    out = op(["aaa\n", "bbb\n"])
    assert out == ["AAA\n"]


def test_search_replace_applied_to_base_only():
    """RNG forced so parents[0] is base; verify edits applied to it."""
    rng = random.Random(0)
    rng.random = lambda: 0.0  # type: ignore[method-assign]
    llm = _StubLLM([_block("aaa", "AAA")])
    op = LLMSearchReplaceCrossover(llm, _RecordingBuilder(), rng=rng)
    # First random.random() is the prob gate, second picks base/donor.
    # With our lambda always 0.0: prob gate (0.0 <= 1.0) passes, then 0.0 < 0.5 -> base=parents[0].
    out = op(["aaa\nshared\n", "aaa\ndonor\n"])
    assert out == ["AAA\nshared\n"]


def test_validator_rejects_then_accepts():
    state = {"calls": 0}

    def validator(_: str) -> ValidationResult:
        state["calls"] += 1
        if state["calls"] == 1:
            return ValidationResult(ok=False, stage="run", feedback="bad")
        return ValidationResult(ok=True)

    rng = random.Random(0)
    rng.random = lambda: 0.0  # type: ignore[method-assign]
    llm = _StubLLM([_block("aaa", "AAA"), _block("AAA", "ZZZ")])
    op = LLMSearchReplaceCrossover(
        llm,
        _RecordingBuilder(),
        validator=validator,
        max_retries=2,
        rng=rng,
    )
    out = op(["aaa\n", "bbb\n"])
    assert out == ["ZZZ\n"]


def test_returns_empty_when_retries_exhausted():
    llm = _StubLLM(["nope", "nope", "nope"])
    op = LLMSearchReplaceCrossover(
        _StubLLM(["nope", "nope", "nope"]), _RecordingBuilder(), max_retries=2
    )
    op.llm = llm
    assert op(["a", "b"]) == []


def test_invalid_probability_raises():
    with pytest.raises(ValueError):
        LLMSearchReplaceCrossover(
            _StubLLM([]), _RecordingBuilder(), crossover_probability=-0.1
        )


def test_prompt_builder_receives_base_and_donor():
    builder = _RecordingBuilder()
    rng = random.Random(0)
    rng.random = lambda: 0.0  # type: ignore[method-assign]
    llm = _StubLLM([_block("aaa", "AAA")])
    op = LLMSearchReplaceCrossover(llm, builder, rng=rng)
    op(["aaa\n", "donor-content\n"])
    assert builder.last_args == ("aaa\n", "donor-content\n")
