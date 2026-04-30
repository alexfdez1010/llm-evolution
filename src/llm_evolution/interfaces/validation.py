"""Instance validation protocol used by self-repairing LLM operators."""

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@dataclass(slots=True)
class ValidationResult:
    """Result of validating a candidate individual.

    Attributes:
        ok: Whether the candidate passed validation.
        stage: Short tag for the failing stage (e.g. "compile", "runtime").
        feedback: Human-readable detail surfaced back to the LLM on retry.
    """

    ok: bool
    stage: str = ""
    feedback: str = ""

    def as_feedback(self) -> str:
        """Format the result as a single feedback string."""
        if self.ok:
            return ""
        if self.stage:
            return f"[{self.stage}] {self.feedback}".strip()
        return self.feedback


@runtime_checkable
class InstanceValidator(Protocol[T]):
    """Validate a candidate individual produced by an LLM operator.

    Implementations may compile, run, lint, or otherwise inspect the
    candidate and return a ``ValidationResult``.  The default
    ``AlwaysValidValidator`` accepts every candidate — useful when the
    surrounding evolutionary loop does its own evaluation.
    """

    def __call__(self, instance: T) -> ValidationResult:
        """Validate *instance* and return the outcome."""
        ...


class AlwaysValidValidator:
    """Validator that accepts every candidate without inspection."""

    def __call__(self, instance: T) -> ValidationResult:  # noqa: ARG002
        return ValidationResult(ok=True)
