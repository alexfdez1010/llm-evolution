"""Prompt-builder protocols for LLM-driven mutation and crossover.

Decoupling prompt construction from the operator lets callers plug in
domain-specific guidance (RVV intrinsics, CUDA, Python kernels, …)
without re-implementing the search/replace + repair loop.
"""

from typing import Protocol, runtime_checkable

from llm_evolution.ai.interfaces.llm import Message


@runtime_checkable
class MutationPromptBuilder(Protocol):
    """Build the LLM messages for a single mutation request."""

    def __call__(self, instance: str) -> list[Message]:
        """Return the messages prompting the LLM to mutate *instance*."""
        ...


@runtime_checkable
class CrossoverPromptBuilder(Protocol):
    """Build the LLM messages for a single crossover request."""

    def __call__(self, base: str, donor: str) -> list[Message]:
        """Return the messages prompting the LLM to combine two parents.

        The returned blocks must edit ``base`` (not ``donor``).
        """
        ...


@runtime_checkable
class RepairPromptBuilder(Protocol):
    """Build a follow-up message that asks the LLM to fix a failed edit."""

    def __call__(self, current_code: str, error_message: str) -> str:
        """Return the user-content string for a repair turn."""
        ...
