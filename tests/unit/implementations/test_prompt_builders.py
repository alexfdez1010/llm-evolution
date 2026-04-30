"""Unit tests for default prompt builders."""

import random

import pytest

from llm_evolution.implementations.prompt_builders import (
    GenericCrossoverPromptBuilder,
    Technique,
    TechniqueMutationPromptBuilder,
)
from llm_evolution.interfaces.prompt_builder import (
    CrossoverPromptBuilder,
    MutationPromptBuilder,
)


_TECHS = [
    Technique(name="loop unrolling", description="Unroll inner loops"),
    Technique(name="cache blocking", description="Tile loops for cache reuse"),
]


def test_mutation_builder_satisfies_protocol():
    b = TechniqueMutationPromptBuilder("Python kernel", _TECHS)
    assert isinstance(b, MutationPromptBuilder)


def test_mutation_builder_includes_technique_in_user_prompt():
    rng = random.Random(0)
    b = TechniqueMutationPromptBuilder("Python kernel", _TECHS, rng=rng)
    msgs = b("def f(): pass\n")
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    user_text = msgs[1].content
    assert any(t.name in user_text for t in _TECHS)
    assert "def f(): pass" in user_text


def test_mutation_builder_uses_code_fence_lang():
    b = TechniqueMutationPromptBuilder(
        "Python kernel", _TECHS, code_fence_lang="python"
    )
    msgs = b("x = 1\n")
    assert "```python" in msgs[1].content


def test_mutation_builder_extra_rules_appended():
    b = TechniqueMutationPromptBuilder(
        "C code", _TECHS, extra_rules="Never use malloc."
    )
    msgs = b("int main() {}")
    assert "Never use malloc." in msgs[0].content


def test_mutation_builder_empty_techniques_raises():
    with pytest.raises(ValueError):
        TechniqueMutationPromptBuilder("X", [])


def test_mutation_builder_custom_picker():
    picked = _TECHS[1]
    b = TechniqueMutationPromptBuilder(
        "Python kernel", _TECHS, technique_picker=lambda _: picked
    )
    msgs = b("x = 1\n")
    assert picked.name in msgs[1].content
    assert _TECHS[0].name not in msgs[1].content


def test_crossover_builder_satisfies_protocol():
    b = GenericCrossoverPromptBuilder("Python kernel")
    assert isinstance(b, CrossoverPromptBuilder)


def test_crossover_builder_renders_base_and_donor():
    b = GenericCrossoverPromptBuilder("Python kernel", code_fence_lang="python")
    msgs = b("BASE-CODE\n", "DONOR-CODE\n")
    user = msgs[1].content
    assert "BASE" in user
    assert "DONOR" in user
    assert "BASE-CODE" in user
    assert "DONOR-CODE" in user
    assert "```python" in user


def test_crossover_builder_numbers_base_lines_by_default():
    b = GenericCrossoverPromptBuilder("X")
    msgs = b("aaa\nbbb\n", "donor\n")
    assert "1|" in msgs[1].content


def test_crossover_builder_can_disable_line_numbers():
    b = GenericCrossoverPromptBuilder("X", number_base_lines=False)
    msgs = b("aaa\nbbb\n", "donor\n")
    assert "1|" not in msgs[1].content


def test_crossover_builder_extra_rules_appended():
    b = GenericCrossoverPromptBuilder("X", extra_rules="Preserve API.")
    msgs = b("a", "b")
    assert "Preserve API." in msgs[0].content
