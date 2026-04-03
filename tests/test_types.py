from edge_agent.types import (
    AgentStep,
    Message,
    RunResult,
    ToolCall,
    ToolCallRecord,
    ToolResult,
)


class TestToolCall:
    def test_create_with_all_fields(self):
        tc = ToolCall(name="search", arguments={"q": "hello"}, id="abc")
        assert tc.name == "search"
        assert tc.arguments == {"q": "hello"}
        assert tc.id == "abc"

    def test_id_defaults_to_none(self):
        tc = ToolCall(name="search", arguments={})
        assert tc.id is None


class TestToolResult:
    def test_create_with_all_fields(self):
        tr = ToolResult(content="42", tool_call_id="abc")
        assert tr.content == "42"
        assert tr.tool_call_id == "abc"

    def test_tool_call_id_defaults_to_none(self):
        tr = ToolResult(content="result")
        assert tr.tool_call_id is None


class TestMessage:
    def test_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_result is None

    def test_assistant_text_message(self):
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"

    def test_assistant_tool_call_message(self):
        tc = ToolCall(name="greet", arguments={"name": "World"}, id="1")
        msg = Message(role="assistant", tool_calls=[tc])
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "greet"

    def test_tool_result_message(self):
        tr = ToolResult(content="done", tool_call_id="1")
        msg = Message(role="tool", content="done", tool_result=tr)
        assert msg.role == "tool"
        assert msg.tool_result is not None
        assert msg.tool_result.content == "done"

    def test_system_message(self):
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."


class TestToolCallRecord:
    def test_create_with_all_fields(self):
        rec = ToolCallRecord(
            name="search", arguments={"q": "hello"}, result="found it", duration_ms=1.5,
        )
        assert rec.name == "search"
        assert rec.arguments == {"q": "hello"}
        assert rec.result == "found it"
        assert rec.duration_ms == 1.5


class TestAgentStep:
    def test_create_with_all_fields(self):
        record = ToolCallRecord(
            name="calc", arguments={"x": 1}, result="2", duration_ms=0.3,
        )
        step = AgentStep(
            agent_name="math-1",
            agent_type="agent",
            tools_used=[record],
            output="The answer is 2.",
            turns=2,
        )
        assert step.agent_name == "math-1"
        assert step.agent_type == "agent"
        assert len(step.tools_used) == 1
        assert step.output == "The answer is 2."
        assert step.turns == 2


class TestRunResult:
    def test_str_returns_output(self):
        result = RunResult(output="hello", steps=[])
        assert str(result) == "hello"

    def test_parsed_defaults_to_none(self):
        result = RunResult(output="hello", steps=[])
        assert result.parsed is None

    def test_parsed_can_be_set(self):
        result = RunResult(output='{"x": 1}', steps=[], parsed={"x": 1})
        assert result.parsed == {"x": 1}

    def test_steps_populated(self):
        step = AgentStep(
            agent_name="a", agent_type="agent", tools_used=[], output="hi", turns=1,
        )
        result = RunResult(output="hi", steps=[step])
        assert len(result.steps) == 1
        assert result.steps[0].agent_name == "a"
