from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from edge_agent import Agent, tool
from edge_agent.types import Message, RunResult, ToolCall

from tests.conftest import MockProvider


class TestAgentTextResponse:
    def test_returns_run_result(self, greet_tool):
        responses = [Message(role="assistant", content="Hi!")]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Be helpful.",
            provider=provider,
            tools=[greet_tool],
        )

        result = agent.run("Hello")
        assert isinstance(result, RunResult)
        assert result.output == "Hi!"
        assert str(result) == "Hi!"

    def test_provider_receives_system_and_user_messages(self, greet_tool):
        responses = [Message(role="assistant", content="ok")]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="System prompt.",
            provider=provider,
            tools=[greet_tool],
        )

        agent.run("User prompt.")

        messages_sent = provider.call_log[0][0]
        assert messages_sent[0].role == "system"
        assert messages_sent[0].content == "System prompt."
        assert messages_sent[1].role == "user"
        assert messages_sent[1].content == "User prompt."


class TestAgentToolCalling:
    def test_single_tool_call(self, greet_tool):
        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="greet", arguments={"name": "Alice"}, id="c1")
                ],
            ),
            Message(role="assistant", content="I greeted Alice for you."),
        ]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Greet people.",
            provider=provider,
            tools=[greet_tool],
        )

        result = agent.run("Say hi to Alice")
        assert result.output == "I greeted Alice for you."
        assert len(provider.call_log) == 2

        second_call_messages = provider.call_log[1][0]
        tool_result_msg = second_call_messages[-1]
        assert tool_result_msg.role == "tool"
        assert tool_result_msg.tool_result.content == "Hello, Alice."

    def test_tool_call_with_optional_param(self, greet_tool):
        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="greet",
                        arguments={"name": "Bob", "excited": True},
                        id="c1",
                    )
                ],
            ),
            Message(role="assistant", content="Done!"),
        ]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Greet people.",
            provider=provider,
            tools=[greet_tool],
        )

        result = agent.run("Excitedly greet Bob")
        assert result.output == "Done!"

        second_call_messages = provider.call_log[1][0]
        tool_result_msg = second_call_messages[-1]
        assert tool_result_msg.tool_result.content == "Hello, Bob!"


class TestAgentToolChaining:
    def test_chains_two_tools(self, greet_tool, add_tool):
        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="add", arguments={"a": 2, "b": 3}, id="c1")
                ],
            ),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="greet",
                        arguments={"name": "Result is 5"},
                        id="c2",
                    )
                ],
            ),
            Message(role="assistant", content="I added 2+3=5 and greeted."),
        ]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Do math and greet.",
            provider=provider,
            tools=[greet_tool, add_tool],
        )

        result = agent.run("Add 2+3 then greet with the result")
        assert result.output == "I added 2+3=5 and greeted."
        assert len(provider.call_log) == 3

    def test_max_turns_stops_loop(self, greet_tool):
        infinite_tool_calls = Message(
            role="assistant",
            tool_calls=[
                ToolCall(name="greet", arguments={"name": "loop"}, id="c1")
            ],
        )
        responses = [infinite_tool_calls] * 20
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Loop forever.",
            provider=provider,
            tools=[greet_tool],
        )

        agent.run("loop", max_turns=3)
        assert len(provider.call_log) == 3


class TestAgentToolScoping:
    def test_agents_have_separate_tools(self, greet_tool, add_tool):
        responses_a = [Message(role="assistant", content="I can greet.")]
        responses_b = [Message(role="assistant", content="I can add.")]

        agent_a = Agent(
            name="greeter",
            instructions="Greet.",
            provider=MockProvider(responses_a),
            tools=[greet_tool],
        )
        agent_b = Agent(
            name="adder",
            instructions="Add.",
            provider=MockProvider(responses_b),
            tools=[add_tool],
        )

        assert "greet" in agent_a.tools
        assert "add" not in agent_a.tools
        assert "add" in agent_b.tools
        assert "greet" not in agent_b.tools


class TestAgentErrorHandling:
    def test_unknown_tool_returns_error(self, greet_tool):
        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="nonexistent", arguments={}, id="c1")
                ],
            ),
            Message(role="assistant", content="Handled error."),
        ]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Test.",
            provider=provider,
            tools=[greet_tool],
        )

        result = agent.run("call unknown")
        assert result.output == "Handled error."

        tool_result_msg = provider.call_log[1][0][-1]
        assert "Unknown tool" in tool_result_msg.tool_result.content

    def test_tool_exception_returns_error_string(self, failing_tool_fixture):
        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="failing_tool",
                        arguments={"x": "test"},
                        id="c1",
                    )
                ],
            ),
            Message(role="assistant", content="Handled."),
        ]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Test.",
            provider=provider,
            tools=[failing_tool_fixture],
        )

        result = agent.run("fail")
        assert result.output == "Handled."

        tool_result_msg = provider.call_log[1][0][-1]
        assert "Error:" in tool_result_msg.tool_result.content
        assert "ValueError" in tool_result_msg.tool_result.content
        assert "boom" not in tool_result_msg.tool_result.content


class TestAgentDefaults:
    def test_name_auto_generated_when_omitted(self):
        provider = MockProvider([Message(role="assistant", content="ok")])
        agent = Agent(instructions="Hi.", provider=provider)
        assert agent.name.startswith("agent-")

    def test_auto_names_are_unique(self):
        provider = MockProvider([Message(role="assistant", content="ok")])
        a1 = Agent(instructions="Hi.", provider=provider)
        a2 = Agent(instructions="Hi.", provider=provider)
        assert a1.name != a2.name

    def test_explicit_name_preserved(self):
        provider = MockProvider([Message(role="assistant", content="ok")])
        agent = Agent(name="my-agent", instructions="Hi.", provider=provider)
        assert agent.name == "my-agent"

    def test_default_instructions(self):
        provider = MockProvider([Message(role="assistant", content="ok")])
        agent = Agent(provider=provider)
        assert agent.instructions == "You are a helpful assistant."

    @patch("edge_agent.agent.Agent._default_provider")
    def test_auto_creates_provider_when_omitted(self, mock_default):
        mock_default.return_value = MockProvider(
            [Message(role="assistant", content="ok")]
        )
        agent = Agent(instructions="Hi.")
        mock_default.assert_called_once()
        assert agent.provider is mock_default.return_value


class TestAgentDuplicateTools:
    def test_duplicate_tool_names_raises(self):
        @tool
        def my_tool(x: str) -> str:
            """First tool."""
            return x

        @tool
        def my_tool(x: str) -> str:  # noqa: F811
            """Second tool with same name."""
            return x

        responses = [Message(role="assistant", content="ok")]
        provider = MockProvider(responses)

        with pytest.raises(ValueError, match="Duplicate tool name: 'my_tool'"):
            Agent(
                name="test",
                instructions="Test.",
                provider=provider,
                tools=[my_tool, my_tool],
            )


class TestFileBasedInstructions:
    def test_path_reads_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
        ) as f:
            f.write("Instructions from file.")
            tmp_path = Path(f.name)

        try:
            provider = MockProvider([Message(role="assistant", content="ok")])
            agent = Agent(
                name="test",
                instructions=tmp_path,
                provider=provider,
            )
            assert agent.instructions == "Instructions from file."
        finally:
            tmp_path.unlink()

    def test_string_instructions_unchanged(self):
        provider = MockProvider([Message(role="assistant", content="ok")])
        agent = Agent(
            name="test",
            instructions="Plain string.",
            provider=provider,
        )
        assert agent.instructions == "Plain string."


class TestTemplateVars:
    def test_system_message_rendered(self):
        responses = [Message(role="assistant", content="ok")]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Hello, {{name}}! Today is {{currentDate}}.",
            provider=provider,
        )

        agent.run("hi", template_vars={"name": "Alice"})

        system_msg = provider.call_log[0][0][0]
        assert system_msg.role == "system"
        assert "Hello, Alice!" in system_msg.content
        assert "{{name}}" not in system_msg.content

    def test_no_template_vars_leaves_unknown_placeholders(self):
        responses = [Message(role="assistant", content="ok")]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Hello, {{name}}!",
            provider=provider,
        )

        agent.run("hi")

        system_msg = provider.call_log[0][0][0]
        assert system_msg.content == "Hello, {{name}}!"


class TestStructuredOutput:
    def test_run_with_output_type(self):
        @dataclass
        class City:
            name: str
            population: int

        json_body = json.dumps({"name": "Tokyo", "population": 14000000})
        responses = [Message(role="assistant", content=json_body)]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Geography expert.",
            provider=provider,
        )

        result = agent.run("Tell me about Tokyo.", output_type=City)
        assert isinstance(result, RunResult)
        assert isinstance(result.parsed, City)
        assert result.parsed.name == "Tokyo"
        assert result.parsed.population == 14000000

    def test_run_without_output_type_returns_run_result(self):
        responses = [Message(role="assistant", content="Just text.")]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Be helpful.",
            provider=provider,
        )

        result = agent.run("hi")
        assert isinstance(result, RunResult)
        assert result.output == "Just text."
        assert result.parsed is None

    def test_output_schema_passed_to_provider(self):
        @dataclass
        class Item:
            title: str

        json_body = json.dumps({"title": "Test"})
        responses = [Message(role="assistant", content=json_body)]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Test.",
            provider=provider,
        )

        agent.run("go", output_type=Item)

        schema_sent = provider.call_log[0][2]
        assert schema_sent is not None
        assert schema_sent["type"] == "object"
        assert "title" in schema_sent["properties"]

    def test_tools_and_output_type_sent_together(self, greet_tool):
        @dataclass
        class Result:
            answer: str

        json_body = json.dumps({"answer": "yes"})
        responses = [Message(role="assistant", content=json_body)]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Test.",
            provider=provider,
            tools=[greet_tool],
        )

        agent.run("go", output_type=Result)

        tools_sent = provider.call_log[0][1]
        schema_sent = provider.call_log[0][2]
        assert tools_sent is not None
        assert schema_sent is not None

    def test_class_level_output_type(self):
        @dataclass
        class City:
            name: str

        json_body = json.dumps({"name": "Tokyo"})
        responses = [Message(role="assistant", content=json_body)]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Test.",
            provider=provider,
            output_type=City,
        )

        result = agent.run("go")
        assert isinstance(result.parsed, City)
        assert result.parsed.name == "Tokyo"

    def test_run_output_type_overrides_class_level(self):
        @dataclass
        class Default:
            x: str

        @dataclass
        class Override:
            y: int

        json_body = json.dumps({"y": 42})
        responses = [Message(role="assistant", content=json_body)]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Test.",
            provider=provider,
            output_type=Default,
        )

        result = agent.run("go", output_type=Override)
        assert isinstance(result.parsed, Override)
        assert result.parsed.y == 42

    def test_output_type_with_tool_calls(self, greet_tool):
        @dataclass
        class Result:
            message: str

        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="greet", arguments={"name": "Alice"}, id="c1")
                ],
            ),
            Message(
                role="assistant",
                content=json.dumps({"message": "Hello, Alice."}),
            ),
        ]
        provider = MockProvider(responses)

        agent = Agent(
            name="test",
            instructions="Greet then respond.",
            provider=provider,
            tools=[greet_tool],
            output_type=Result,
        )

        result = agent.run("Greet Alice")
        assert isinstance(result.parsed, Result)
        assert result.parsed.message == "Hello, Alice."


class TestRunResultTracing:
    def test_steps_populated_for_simple_run(self, greet_tool):
        responses = [Message(role="assistant", content="Hi!")]
        provider = MockProvider(responses)

        agent = Agent(
            name="test-agent",
            instructions="Be helpful.",
            provider=provider,
            tools=[greet_tool],
        )

        result = agent.run("Hello")
        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.agent_name == "test-agent"
        assert step.agent_type == "agent"
        assert step.output == "Hi!"
        assert step.turns == 1
        assert step.tools_used == []

    def test_tool_calls_recorded(self, greet_tool):
        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="greet", arguments={"name": "Alice"}, id="c1")
                ],
            ),
            Message(role="assistant", content="Done."),
        ]
        provider = MockProvider(responses)

        agent = Agent(
            name="test-agent",
            instructions="Greet.",
            provider=provider,
            tools=[greet_tool],
        )

        result = agent.run("Greet Alice")
        assert len(result.steps) == 1
        step = result.steps[0]
        assert step.turns == 2
        assert len(step.tools_used) == 1
        rec = step.tools_used[0]
        assert rec.name == "greet"
        assert rec.arguments == {"name": "Alice"}
        assert rec.result == "Hello, Alice."
        assert rec.duration_ms >= 0.0

    def test_multiple_tool_calls_recorded(self, greet_tool, add_tool):
        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="add", arguments={"a": 1, "b": 2}, id="c1"),
                ],
            ),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="greet", arguments={"name": "Bob"}, id="c2"),
                ],
            ),
            Message(role="assistant", content="All done."),
        ]
        provider = MockProvider(responses)

        agent = Agent(
            name="test-agent",
            instructions="Do stuff.",
            provider=provider,
            tools=[greet_tool, add_tool],
        )

        result = agent.run("go")
        step = result.steps[0]
        assert len(step.tools_used) == 2
        assert step.tools_used[0].name == "add"
        assert step.tools_used[0].result == "3"
        assert step.tools_used[1].name == "greet"
        assert step.tools_used[1].result == "Hello, Bob."

    def test_structured_output_in_parsed(self):
        @dataclass
        class Answer:
            value: int

        json_body = json.dumps({"value": 42})
        responses = [Message(role="assistant", content=json_body)]
        provider = MockProvider(responses)

        agent = Agent(name="test", instructions="x", provider=provider)
        result = agent.run("go", output_type=Answer)
        assert result.output == json_body
        assert isinstance(result.parsed, Answer)
        assert result.parsed.value == 42
