"""Search/replace-based mutation driven by an LLM."""

from __future__ import annotations

import logging
import random

from llm_evolution.ai.interfaces.llm import LLM
from llm_evolution.implementations.search_replace_loop import (
    default_repair_builder,
    run_edit_loop,
)
from llm_evolution.interfaces.mutation import Mutation
from llm_evolution.interfaces.prompt_builder import (
    MutationPromptBuilder,
    RepairPromptBuilder,
)
from llm_evolution.interfaces.validation import InstanceValidator

logger = logging.getLogger(__name__)


class LLMSearchReplaceMutation(Mutation[str]):
    """Mutate a string individual via LLM-emitted search/replace blocks.

    Flexibility hooks:
        * ``prompt_builder``: domain-specific instructions per mutation.
        * ``validator``: optional pre-acceptance check (compile/test/lint).
        * ``repair_builder``: how to phrase retry feedback.
        * ``mutation_probability``: chance the mutation runs at all.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_builder: MutationPromptBuilder,
        *,
        validator: InstanceValidator[str] | None = None,
        repair_builder: RepairPromptBuilder | None = None,
        mutation_probability: float = 1.0,
        max_retries: int = 2,
        rng: random.Random | None = None,
    ):
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in [0, 1]")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.validator = validator
        self.repair_builder = repair_builder or default_repair_builder()
        self.mutation_probability = mutation_probability
        self.max_retries = max_retries
        self._rng = rng or random.Random()

    def __call__(self, instance: str) -> str | None:
        if self._rng.random() > self.mutation_probability:
            return None

        messages = self.prompt_builder(instance)
        result = run_edit_loop(
            self.llm,
            messages,
            instance,
            validator=self.validator,
            repair_builder=self.repair_builder,
            max_retries=self.max_retries,
        )
        if result.code is None:
            logger.info("Mutation failed: %s", result.error[:120])
            return None
        return result.code
