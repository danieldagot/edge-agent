from __future__ import annotations

import pytest

from edge_agent import tool
from edge_agent.providers.base import Provider
from edge_agent.tool import Tool
from edge_agent.types import Message, ToolCall


@tool
def greet(name: str, excited: bool = False) -> str:
    """Greet someone by name."""
    return f"Hello, {name}{'!' if excited else '.'}"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def failing_tool(x: str) -> str:
    """A tool that always raises."""
    raise ValueError(f"boom: {x}")


class MockProvider(Provider):
    """Provider that returns pre-configured responses in sequence."""

    def __init__(self, responses: list[Message]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.call_log: list[
            tuple[list[Message], list[Tool] | None, dict[str, object] | None]
        ] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        output_schema: dict[str, object] | None = None,
    ) -> Message:
        self.call_log.append((list(messages), tools, output_schema))
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


@pytest.fixture
def greet_tool() -> Tool:
    return greet


@pytest.fixture
def add_tool() -> Tool:
    return add


@pytest.fixture
def failing_tool_fixture() -> Tool:
    return failing_tool


@pytest.fixture
def mock_text_response() -> Message:
    return Message(role="assistant", content="Hello there!")


@pytest.fixture
def mock_tool_call_response() -> Message:
    return Message(
        role="assistant",
        tool_calls=[
            ToolCall(name="greet", arguments={"name": "World"}, id="call_1")
        ],
    )
