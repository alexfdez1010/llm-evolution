from typing import Protocol, TypeVar, List, runtime_checkable

T = TypeVar("T")

@runtime_checkable
class Crossover(Protocol[T]):
    """Protocol for crossover operations in evolutionary algorithms."""
    def __call__(self, parents: List[T]) -> List[T]:
        """Combine parents to create offspring."""
        ...

def crossover_fn(fn):
    """Decorator to convert a function into a Crossover protocol implementation."""
    class Wrapper:
        def __init__(self, func):
            self.func = func
        def __call__(self, parents: List[T]) -> List[T]:
            return self.func(parents)
    return Wrapper(fn)
