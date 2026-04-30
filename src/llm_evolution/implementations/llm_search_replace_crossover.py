"""Search/replace-based crossover driven by an LLM."""

from __future__ import annotations

import logging
import random

from llm_evolution.ai.interfaces.llm import LLM
from llm_evolution.implementations.search_replace_loop import (
    default_repair_builder,
    run_edit_loop,
)
from llm_evolution.interfaces.crossover import Crossover
from llm_evolution.interfaces.prompt_builder import (
    CrossoverPromptBuilder,
    RepairPromptBuilder,
)
from llm_evolution.interfaces.validation import InstanceValidator

logger = logging.getLogger(__name__)


class LLMSearchReplaceCrossover(Crossover[str]):
    """Combine two parents via LLM-emitted search/replace blocks on the base.

    Picks one parent as the *base* (target of the edits) and another as the
    *donor* (source of features).  The LLM is asked to lift one or two
    features from donor into base via small, focused search/replace blocks.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_builder: CrossoverPromptBuilder,
        *,
        validator: InstanceValidator[str] | None = None,
        repair_builder: RepairPromptBuilder | None = None,
        crossover_probability: float = 1.0,
        max_retries: int = 2,
        rng: random.Random | None = None,
    ):
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be in [0, 1]")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.validator = validator
        self.repair_builder = repair_builder or default_repair_builder()
        self.crossover_probability = crossover_probability
        self.max_retries = max_retries
        self._rng = rng or random.Random()

    def __call__(self, parents: list[str]) -> list[str]:
        if len(parents) < 2:
            return []
        if self._rng.random() > self.crossover_probability:
            return []

        if self._rng.random() < 0.5:
            base, donor = parents[0], parents[1]
        else:
            base, donor = parents[1], parents[0]

        messages = self.prompt_builder(base, donor)
        result = run_edit_loop(
            self.llm,
            messages,
            base,
            validator=self.validator,
            repair_builder=self.repair_builder,
            max_retries=self.max_retries,
        )
        if result.code is None:
            logger.info("Crossover failed: %s", result.error[:120])
            return []
        return [result.code]
