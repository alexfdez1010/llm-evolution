import random
from typing import TypeVar

from ..interfaces.selection import Selection

T = TypeVar("T")


class TournamentSelection(Selection[T]):
    """
    Tournament Selection strategy.
    Selects survivors by holding tournaments among randomly selected individuals
    from the combined pool of current population and offspring.
    """

    def __init__(self, tournament_size: int = 3, population_size: int = 100):
        """
        Initialize the TournamentSelection strategy.

        Args:
            tournament_size: Number of individuals participating in each tournament.
            population_size: The desired number of individuals in the next generation.
        """
        self.tournament_size = tournament_size
        self.population_size = population_size

    def __call__(
        self, population: list[T], offspring: list[T], fitness_scores: list[float]
    ) -> list[T]:
        """
        Selects survivors using tournament selection.

        Args:
            population: The current population.
            offspring: The offspring generated.
            fitness_scores: Fitness scores for (population + offspring).

        Returns:
            The selected survivors for the next generation.
        """
        pool = population + offspring
        if len(pool) != len(fitness_scores):
            raise ValueError(
                f"Mismatch between pool size ({len(pool)}) and fitness scores ({len(fitness_scores)})"
            )

        if not pool:
            return []

        # Combine individuals with their fitness
        candidates = list(zip(pool, fitness_scores))
        selected: list[T] = []

        for _ in range(self.population_size):
            # Sample candidates for the tournament
            # Use replacement=True conceptually for the population selection, 
            # but random.sample is without replacement for the specific tournament batch.
            # We sample from the *entire pool* each time.
            
            # Note: random.sample requires population sequence.
            # If pool is smaller than tournament_size, take the whole pool.
            tournament_candidates = random.sample(
                candidates, min(len(candidates), self.tournament_size)
            )
            
            # The winner is the one with the highest fitness
            winner = max(tournament_candidates, key=lambda item: item[1])[0]
            selected.append(winner)

        return selected
