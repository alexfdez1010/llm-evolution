"""Elitist survivor selection implementation."""

from typing import TypeVar

from llm_evolution.interfaces.selection import Selection

T = TypeVar("T")


class ElitistSelection(Selection[T]):
    """
    Select top-scoring unique individuals from population and offspring.

    This implementation combines `population` and `offspring`, sorts by descending
    fitness, keeps only unique individuals, and returns up to `max_size`.
    """

    def __init__(self, max_size: int):
        """
        Initialize an elitist selector.

        Args:
            max_size: Maximum number of survivors to return.

        Raises:
            ValueError: If `max_size` is not positive.
        """
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")
        self.max_size = max_size

    def __call__(
        self,
        population: list[T],
        offspring: list[T],
        fitness_scores: list[float],
    ) -> list[T]:
        """
        Select survivors based on descending fitness with deduplication.

        Args:
            population: Current population.
            offspring: Newly generated individuals.
            fitness_scores: Scores for `population + offspring`.

        Returns:
            list[T]: Survivors, capped by `max_size`.

        Raises:
            ValueError: If fitness score count does not match individuals.
        """
        combined = population + offspring
        if len(combined) != len(fitness_scores):
            raise ValueError(
                f"Mismatch: {len(combined)} individuals vs {len(fitness_scores)} scores"
            )

        indexed_scores = sorted(
            enumerate(fitness_scores), key=lambda item: item[1], reverse=True
        )
        survivors: list[T] = []
        seen_keys: set[int] = set()

        for index, _score in indexed_scores:
            candidate = combined[index]
            try:
                candidate_key = hash(candidate)
            except TypeError:
                # Unhashable instances are deduplicated by identity.
                candidate_key = id(candidate)
            if candidate_key in seen_keys:
                continue
            seen_keys.add(candidate_key)
            survivors.append(candidate)
            if len(survivors) >= self.max_size:
                break

        if survivors:
            return survivors
        if population:
            return [population[0]]
        return []
