"""Shared LLM + search/replace + validation retry loop.

Used by ``LLMSearchReplaceMutation`` and ``LLMSearchReplaceCrossover`` to
generate, parse, apply, and validate edits with self-repair turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from llm_evolution.ai.interfaces.llm import LLM, Message
from llm_evolution.interfaces.prompt_builder import RepairPromptBuilder
from llm_evolution.interfaces.validation import (
    AlwaysValidValidator,
    InstanceValidator,
)
from llm_evolution.implementations.utils.search_replace import (
    apply_search_replace,
    extract_search_replace,
    search_replace_error_feedback,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EditLoopResult:
    """Outcome of a search/replace edit loop."""

    code: str | None
    error: str = ""


def default_repair_builder(file_name: str = "file") -> RepairPromptBuilder:
    """Repair builder that surfaces the failure with line-numbered context."""

    def _build(current_code: str, error_message: str) -> str:
        return search_replace_error_feedback(file_name, current_code, error_message)

    return _build


def run_edit_loop(
    llm: LLM,
    messages: list[Message],
    current_code: str,
    *,
    validator: InstanceValidator[str] | None = None,
    repair_builder: RepairPromptBuilder | None = None,
    max_retries: int = 2,
) -> EditLoopResult:
    """Run the LLM -> search/replace -> validate loop with repair retries.

    Returns the first candidate that parses, applies, and validates ok.
    """
    validator = validator or AlwaysValidValidator()
    repair_builder = repair_builder or default_repair_builder()

    active_messages = list(messages)
    working_code = current_code
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            response = llm(active_messages)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM generation failed on attempt %d: %s", attempt + 1, exc)
            return EditLoopResult(code=None, error=str(exc))

        blocks = extract_search_replace(response)
        if blocks is None:
            last_error = "Could not extract search/replace blocks from LLM response"
            logger.debug("%s on attempt %d", last_error, attempt + 1)
            if attempt >= max_retries:
                return EditLoopResult(code=None, error=last_error)
            active_messages = active_messages + [
                Message(role="assistant", content=response),
                Message(
                    role="user",
                    content=repair_builder(working_code, last_error),
                ),
            ]
            continue

        try:
            candidate = apply_search_replace(working_code, blocks, strict=False)
        except ValueError as exc:
            last_error = str(exc)
            logger.debug("Apply failed on attempt %d: %s", attempt + 1, exc)
            if attempt >= max_retries:
                return EditLoopResult(code=None, error=last_error)
            active_messages = active_messages + [
                Message(role="assistant", content=response),
                Message(
                    role="user",
                    content=repair_builder(working_code, last_error),
                ),
            ]
            continue

        validation = validator(candidate)
        if validation.ok:
            return EditLoopResult(code=candidate, error="")

        last_error = f"validation failed at {validation.stage}".strip()
        logger.debug("Candidate %s on attempt %d", last_error, attempt + 1)
        if attempt >= max_retries:
            return EditLoopResult(code=None, error=last_error)

        working_code = candidate
        active_messages = active_messages + [
            Message(role="assistant", content=response),
            Message(
                role="user", content=repair_builder(candidate, validation.as_feedback())
            ),
        ]

    return EditLoopResult(code=None, error=last_error)
