from typing import Protocol, runtime_checkable, List


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for Embedding Models."""

    def __call__(self, text: str) -> List[float]:
        """
        Generates an embedding vector for the given text.

        Args:
            text: The input text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        ...


def embedding_model_fn(fn):
    """
    Decorator to convert a function into an EmbeddingModel protocol implementation.

    Args:
        fn: A function that takes a text and returns an embedding vector.

    Returns:
        Wrapper: A class implementing the EmbeddingModel protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, text: str) -> List[float]:
            return self.func(text)

    return Wrapper(fn)
