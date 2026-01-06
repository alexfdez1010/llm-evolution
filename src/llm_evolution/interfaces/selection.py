from typing import Protocol, TypeVar, List, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Selection(Protocol[T]):
    """Protocol for selecting survivors for the next generation."""

    def __call__(
        self, population: List[T], offspring: List[T], fitness_scores: List[float]
    ) -> List[T]:
        """Select individuals to survive to the next generation."""
        ...


def selection_fn(fn):
    """Decorator to convert a function into a Selection protocol implementation."""

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(
            self, population: List[T], offspring: List[T], fitness_scores: List[float]
        ) -> List[T]:
            return self.func(population, offspring, fitness_scores)

    return Wrapper(fn)
