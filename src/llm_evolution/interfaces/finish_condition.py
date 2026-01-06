from typing import Protocol, TypeVar, List, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class FinishCondition(Protocol[T]):
    """Protocol for determining when to finish the evolutionary process."""

    def __call__(
        self, population: List[T], generation: int, fitness_scores: List[float]
    ) -> bool:
        """Return True if the evolutionary process should stop."""
        ...


def finish_condition_fn(fn):
    """Decorator to convert a function into a FinishCondition protocol implementation."""

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(
            self, population: List[T], generation: int, fitness_scores: List[float]
        ) -> bool:
            return self.func(population, generation, fitness_scores)

    return Wrapper(fn)
