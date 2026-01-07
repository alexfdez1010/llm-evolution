from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Mutation(Protocol[T]):
    """Protocol for mutation operations in evolutionary algorithms."""

    def __call__(self, instance: T) -> T | None:
        """
        Apply random variations to a single individual.

        Args:
            instance: The individual instance to be mutated.

        Returns:
            T | None: A new individual resulting from the mutation process,
                       or None if no mutation was performed.
        """
        ...


def mutation_fn(fn):
    """
    Decorator to convert a function into a Mutation protocol implementation.

    Args:
        fn: A function that takes an instance and returns a mutated instance or None.

    Returns:
        Wrapper: A class implementing the Mutation protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, instance: T) -> T | None:
            return self.func(instance)

    return Wrapper(fn)
