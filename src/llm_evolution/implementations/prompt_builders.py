"""Default prompt builders for LLM mutation and crossover.

These are intentionally generic.  Wire your own ``MutationPromptBuilder``
or ``CrossoverPromptBuilder`` for domain-specific guidance (e.g. RVV
intrinsics, CUDA, JAX kernels).  The builders here cover the common case:
edit one text artefact, optionally guided by a randomly-chosen technique.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llm_evolution.ai.interfaces.llm import Message
from llm_evolution.implementations.utils.search_replace import (
    number_lines,
    search_replace_format_example,
)


@dataclass(slots=True)
class Technique:
    """A named, described mutation idea fed into the LLM prompt."""

    name: str
    description: str


def _format_techniques(techniques: Sequence[Technique]) -> str:
    return "\n".join(f"- **{t.name}**: {t.description}" for t in techniques)


class TechniqueMutationPromptBuilder:
    """Mutation prompt that picks one technique per call.

    Args:
        domain: Short tag describing the artefact ("Python kernel", "C
            code", "SQL query"); injected into the system prompt.
        techniques: Catalogue of techniques.  One is sampled each call.
        code_fence_lang: Code-fence language (e.g. ``"python"``, ``"c"``).
        rng: Optional ``random.Random`` for reproducible sampling.
        extra_rules: Extra system-prompt rules appended verbatim.
        technique_picker: Override sampling (default: uniform).
    """

    def __init__(
        self,
        domain: str,
        techniques: Sequence[Technique],
        *,
        code_fence_lang: str = "",
        extra_rules: str = "",
        rng: random.Random | None = None,
        technique_picker: Callable[[Sequence[Technique]], Technique] | None = None,
    ):
        if not techniques:
            raise ValueError("techniques must not be empty")
        self.domain = domain
        self.techniques = list(techniques)
        self.code_fence_lang = code_fence_lang
        self.extra_rules = extra_rules
        self._rng = rng or random.Random()
        self._pick = technique_picker or (lambda ts: self._rng.choice(list(ts)))

    def __call__(self, instance: str) -> list[Message]:
        technique = self._pick(self.techniques)
        system = (
            f"You are an expert improving a {self.domain}.  Apply ONE focused "
            "optimization at a time using minimal search/replace blocks.\n"
            "Hard rules:\n"
            "- Edit ONLY the target artefact.  Public API stays unchanged.\n"
            "- Apply EXACTLY the one technique the user names.  No extra edits.\n"
            "- Output: one short summary line, then minimal search/replace "
            "block(s) — each small, focused, unique in the file.\n"
        )
        if self.extra_rules:
            system = system + "\n" + self.extra_rules + "\n"
        system = system + "\n" + search_replace_format_example()

        user = (
            f"Apply this single optimization:\n\n"
            f"**Technique: {technique.name}**\n"
            f"{technique.description}\n\n"
            f"Current artefact:\n```{self.code_fence_lang}\n{instance}\n```"
        )
        return [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]


class GenericCrossoverPromptBuilder:
    """Crossover prompt that lifts features from donor into base.

    Args:
        domain: Short tag describing the artefact.
        code_fence_lang: Code-fence language.
        extra_rules: Extra system-prompt rules appended verbatim.
        number_base_lines: When ``True`` (default), the base is shown with
            line numbers for easier reference in the LLM response.
    """

    def __init__(
        self,
        domain: str,
        *,
        code_fence_lang: str = "",
        extra_rules: str = "",
        number_base_lines: bool = True,
    ):
        self.domain = domain
        self.code_fence_lang = code_fence_lang
        self.extra_rules = extra_rules
        self.number_base_lines = number_base_lines

    def __call__(self, base: str, donor: str) -> list[Message]:
        system = (
            f"You are an expert improving a {self.domain}.  Combine the BEST "
            "features of a DONOR variant into a BASE variant.\n\n"
            "RULES:\n"
            "- The search/replace blocks MUST apply to the BASE variant.\n"
            "- Pick ONE or TWO best features from the DONOR.\n"
            "- Apply those features via small, focused search/replace blocks.\n"
            "- Do NOT change function signatures or the public API.\n"
            "- CRITICAL: SEARCH text MUST be copied EXACTLY from the BASE, "
            "character for character.  Do NOT use lines from the DONOR.\n"
        )
        if self.extra_rules:
            system = system + "\n" + self.extra_rules + "\n"
        system = system + "\n" + search_replace_format_example()

        rendered_base = number_lines(base) if self.number_base_lines else base
        user = (
            f"Incorporate the best optimizations from the DONOR into the BASE.\n\n"
            f"BASE (apply search/replace to this):\n"
            f"```{self.code_fence_lang}\n{rendered_base}\n```\n\n"
            f"DONOR (take features from this):\n"
            f"```{self.code_fence_lang}\n{donor}\n```\n"
        )
        return [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
