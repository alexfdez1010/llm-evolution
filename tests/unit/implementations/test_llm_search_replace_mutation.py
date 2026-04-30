"""Unit tests for LLMSearchReplaceMutation."""

from __future__ import annotations

import random

import pytest

from llm_evolution.ai.interfaces.llm import Message
from llm_evolution.implementations.llm_search_replace_mutation import (
    LLMSearchReplaceMutation,
)
from llm_evolution.interfaces.mutation import Mutation
from llm_evolution.interfaces.validation import ValidationResult


class _StubLLM:
    """Pop responses from a queue; record received messages."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def __call__(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        return self._responses.pop(0)


def _builder(instance: str) -> list[Message]:
    return [
        Message(role="system", content="mutate"),
        Message(role="user", content=instance),
    ]


def _block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


def test_implements_mutation_protocol():
    op = LLMSearchReplaceMutation(_StubLLM([]), _builder)
    assert isinstance(op, Mutation)


def test_skips_when_probability_zero():
    op = LLMSearchReplaceMutation(_StubLLM([]), _builder, mutation_probability=0.0)
    assert op("instance") is None


def test_returns_modified_instance_on_success():
    llm = _StubLLM([_block("aaa", "AAA")])
    op = LLMSearchReplaceMutation(llm, _builder)
    result = op("aaa\nbbb\n")
    assert result == "AAA\nbbb\n"
    assert len(llm.calls) == 1


def test_retries_when_no_blocks_extracted():
    llm = _StubLLM(["nothing here", _block("aaa", "AAA")])
    op = LLMSearchReplaceMutation(llm, _builder, max_retries=2)
    result = op("aaa\n")
    assert result == "AAA\n"
    assert len(llm.calls) == 2


def test_returns_none_when_retries_exhausted():
    llm = _StubLLM(["junk", "still junk", "nope"])
    op = LLMSearchReplaceMutation(llm, _builder, max_retries=2)
    assert op("aaa\n") is None
    assert len(llm.calls) == 3


def test_retries_when_search_text_missing():
    llm = _StubLLM([_block("nope", "x"), _block("aaa", "AAA")])
    op = LLMSearchReplaceMutation(llm, _builder, max_retries=1)
    assert op("aaa\n") == "AAA\n"


def test_validator_rejects_then_accepts():
    state = {"calls": 0}

    def validator(_: str) -> ValidationResult:
        state["calls"] += 1
        if state["calls"] == 1:
            return ValidationResult(ok=False, stage="lint", feedback="bad")
        return ValidationResult(ok=True)

    llm = _StubLLM([_block("aaa", "AAA"), _block("AAA", "BBB")])
    op = LLMSearchReplaceMutation(llm, _builder, validator=validator, max_retries=2)
    assert op("aaa\n") == "BBB\n"
    assert state["calls"] == 2


def test_validator_failure_returns_none_when_retries_exhausted():
    def validator(_: str) -> ValidationResult:
        return ValidationResult(ok=False, stage="run", feedback="boom")

    llm = _StubLLM([_block("aaa", "AAA"), _block("AAA", "BBB")])
    op = LLMSearchReplaceMutation(llm, _builder, validator=validator, max_retries=1)
    assert op("aaa\n") is None


def test_llm_exception_returns_none():
    class _Bad:
        def __call__(self, _messages):
            raise RuntimeError("network down")

    op = LLMSearchReplaceMutation(_Bad(), _builder)
    assert op("aaa\n") is None


def test_invalid_probability_raises():
    with pytest.raises(ValueError):
        LLMSearchReplaceMutation(_StubLLM([]), _builder, mutation_probability=1.5)


def test_negative_max_retries_raises():
    with pytest.raises(ValueError):
        LLMSearchReplaceMutation(_StubLLM([]), _builder, max_retries=-1)


def test_rng_controls_probability_gate():
    """Deterministic RNG: prob 0.5, force random.random() to return 0.9 -> skip."""
    rng = random.Random(0)
    rng.random = lambda: 0.9  # type: ignore[method-assign]
    op = LLMSearchReplaceMutation(
        _StubLLM([_block("a", "A")]),
        _builder,
        mutation_probability=0.5,
        rng=rng,
    )
    assert op("aaa\n") is None


def test_repair_message_appended_on_retry():
    llm = _StubLLM(["junk", _block("aaa", "AAA")])
    op = LLMSearchReplaceMutation(llm, _builder, max_retries=1)
    op("aaa\n")
    assert len(llm.calls[1]) > len(llm.calls[0])
