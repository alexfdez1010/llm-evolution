from typing import Any
from openai import OpenAI
from llm_evolution.ai.interfaces.llm import LLM, Message


class OpenAILLM(LLM):
    """
    Implementation of the LLM interface using OpenAI-compatible endpoints.

    This implementation works with OpenAI's official API as well as any
    compatible service (e.g., LocalAI, vLLM, Ollama) by configuring the base_url.
    """

    def __init__(
        self,
        model: str,
        api_key: str = "sk-no-key-required",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI LLM.

        Args:
            model: The name of the model to use.
            api_key: The API key for authentication.
            base_url: The base URL of the API endpoint.
                     Allows using any OpenAI-compatible provider.
            **kwargs: Additional arguments passed to the OpenAI client.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def __call__(self, messages: list[Message]) -> str:
        """
        Generates a text response using the OpenAI chat completion endpoint.

        Args:
            messages: A list of message objects.

        Returns:
            str: The generated response text.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
        )
        return response.choices[0].message.content or ""
