from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Protocol for Large Language Models."""

    def __call__(self, messages: list[dict[str, str]]) -> str:
        """
        Generates a text response from a list of messages.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).

        Returns:
            str: The generated text response.
        """
        ...


def llm_fn(fn):
    """
    Decorator to convert a function into an LLM protocol implementation.

    Args:
        fn: A function that takes a list of messages and returns a text response.

    Returns:
        Wrapper: A class implementing the LLM protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, messages: list[dict[str, str]]) -> str:
            return self.func(messages)

    return Wrapper(fn)
