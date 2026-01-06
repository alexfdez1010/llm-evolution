from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T", contravariant=True)


@runtime_checkable
class Evaluation(Protocol[T]):
    """Protocol for evaluating the fitness of an instance."""

    def __call__(self, instance: T) -> float:
        """Calculate and return the fitness score."""
        ...


def evaluation_fn(fn):
    """
    Decorator to convert a function into an Evaluation protocol implementation.

    Args:
        fn: A function that takes an instance and returns its fitness score.

    Returns:
        Wrapper: A class implementing the Evaluation protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, instance: object) -> float:
            return self.func(instance)

    return Wrapper(fn)
