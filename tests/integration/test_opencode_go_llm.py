import os

import pytest
from dotenv import load_dotenv

from llm_evolution.ai.implementations.llm import opencode_go_model
from llm_evolution.ai.interfaces.llm import Message

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("OPENCODE_GO_API_KEY"),
    reason="OPENCODE_GO_API_KEY not set; skipping OpenCode Go integration test.",
)
def test_opencode_go_llm_route_works():
    llm = opencode_go_model("kimi-k2.6")

    response = llm(
        [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Reply with exactly: OK"),
        ]
    )

    assert isinstance(response, str)
    assert response.strip() != ""
