from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Mutation(Protocol[T]):
    """Protocol for mutation operations in evolutionary algorithms."""

    def __call__(self, instance: T) -> T:
        """Apply mutation to an instance."""
        ...


def mutation_fn(fn):
    """
    Decorator to convert a function into a Mutation protocol implementation.

    Args:
        fn: A function that takes an instance and returns a mutated instance.

    Returns:
        Wrapper: A class implementing the Mutation protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, instance: T) -> T:
            return self.func(instance)

    return Wrapper(fn)
