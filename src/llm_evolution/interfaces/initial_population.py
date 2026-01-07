from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class InitialPopulation(Protocol[T]):
    """Protocol for generating the initial population."""

    def __call__(self, size: int) -> list[T]:
        """
        Generate an initial population of individuals.

        Args:
            size: The desired number of individuals in the initial population.

        Returns:
            list[T]: A list of newly generated individuals of the specified size.
        """
        ...


def initial_population_fn(fn):
    """
    Decorator to convert a function into an InitialPopulation protocol implementation.

    Args:
        fn: A function that takes a size and returns an initial population.

    Returns:
        Wrapper: A class implementing the InitialPopulation protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, size: int) -> list[object]:
            return self.func(size)

    return Wrapper(fn)
