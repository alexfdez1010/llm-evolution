from typing import Protocol, TypeVar, List, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Crossover(Protocol[T]):
    """Protocol for crossover operations in evolutionary algorithms."""

    def __call__(self, parents: List[T]) -> List[T]:
        """
        Combine parents to create offspring.

        Args:
            parents: A list of parent individuals selected for reproduction.

        Returns:
            List[T]: A list of new offspring individuals produced by the crossover.
        """
        ...


def crossover_fn(fn):
    """
    Decorator to convert a function into a Crossover protocol implementation.

    Args:
        fn: A function that takes a list of parents and returns a list of offspring.

    Returns:
        Wrapper: A class implementing the Crossover protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, parents: List[T]) -> List[T]:
            return self.func(parents)

    return Wrapper(fn)
