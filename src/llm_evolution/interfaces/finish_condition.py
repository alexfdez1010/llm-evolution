from typing import Protocol, TypeVar, List, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class FinishCondition(Protocol[T]):
    """Protocol for determining when to finish the evolutionary process."""

    def __call__(
        self, population: List[T], generation: int, fitness_scores: List[float]
    ) -> bool:
        """
        Determine whether the evolutionary process should stop based on the current state.

        Args:
            population: The current population of individuals.
            generation: The current generation number.
            fitness_scores: The fitness scores for the current population.

        Returns:
            bool: True if the evolution should stop, False otherwise.
        """
        ...


def finish_condition_fn(fn):
    """
    Decorator to convert a function into a FinishCondition protocol implementation.

    Args:
        fn: A function that takes population, generation, and fitness scores and returns a boolean.

    Returns:
        Wrapper: A class implementing the FinishCondition protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(
            self, population: List[T], generation: int, fitness_scores: List[float]
        ) -> bool:
            return self.func(population, generation, fitness_scores)

    return Wrapper(fn)
