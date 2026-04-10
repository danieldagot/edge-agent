from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from edge_agent.providers.ollama import OllamaProvider
from edge_agent.tool import Tool, tool
from edge_agent.types import Message, ToolCall, ToolResult


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider(model="llama3.2")


def _mock_urlopen(response_body: dict):
    """Return a context-manager mock for urllib.request.urlopen."""
    body = json.dumps(response_body).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _ok_response(content: str = "ok") -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}}
        ]
    }


def _tool_call_response(
    name: str = "get_weather",
    arguments: dict | None = None,
    call_id: str = "call_1",
    content: str | None = None,
) -> dict:
    tc = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}),
        },
    }
    msg: dict = {"role": "assistant", "tool_calls": [tc]}
    if content is not None:
        msg["content"] = content
    return {"choices": [{"message": msg}]}


# ── payload construction ────────────────────────────────────────────────────


class TestBuildPayload:
    def test_system_message(self, provider: OllamaProvider):
        messages = [Message(role="system", content="Be helpful.")]
        payload = provider._build_payload(messages, None)

        assert payload["model"] == "llama3.2"
        assert payload["stream"] is False
        assert len(payload["messages"]) == 1
        assert payload["messages"][0] == {
            "role": "system",
            "content": "Be helpful.",
        }

    def test_user_message(self, provider: OllamaProvider):
        messages = [Message(role="user", content="Hello")]
        payload = provider._build_payload(messages, None)

        assert payload["messages"][0] == {"role": "user", "content": "Hello"}

    def test_assistant_text_message(self, provider: OllamaProvider):
        messages = [Message(role="assistant", content="Hi there")]
        payload = provider._build_payload(messages, None)

        m = payload["messages"][0]
        assert m["role"] == "assistant"
        assert m["content"] == "Hi there"
        assert "tool_calls" not in m

    def test_assistant_tool_call_message(self, provider: OllamaProvider):
        tc = ToolCall(name="search", arguments={"q": "test"}, id="call_1")
        messages = [Message(role="assistant", tool_calls=[tc])]
        payload = provider._build_payload(messages, None)

        m = payload["messages"][0]
        assert m["role"] == "assistant"
        assert len(m["tool_calls"]) == 1
        fn = m["tool_calls"][0]
        assert fn["id"] == "call_1"
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "search"
        assert fn["function"]["arguments"] == '{"q": "test"}'

    def test_assistant_tool_call_none_id_falls_back(self, provider: OllamaProvider):
        tc = ToolCall(name="search", arguments={}, id=None)
        messages = [Message(role="assistant", tool_calls=[tc])]
        payload = provider._build_payload(messages, None)

        assert payload["messages"][0]["tool_calls"][0]["id"] == ""

    def test_tool_result_message(self, provider: OllamaProvider):
        tr = ToolResult(content="42", tool_call_id="call_1", tool_name="calc")
        messages = [Message(role="tool", tool_result=tr)]
        payload = provider._build_payload(messages, None)

        m = payload["messages"][0]
        assert m["role"] == "tool"
        assert m["tool_call_id"] == "call_1"
        assert m["content"] == "42"

    def test_tool_result_none_tool_call_id_falls_back(self, provider: OllamaProvider):
        tr = ToolResult(content="42", tool_call_id=None)
        messages = [Message(role="tool", tool_result=tr)]
        payload = provider._build_payload(messages, None)

        assert payload["messages"][0]["tool_call_id"] == ""

    def test_tools_as_function_declarations(self, provider: OllamaProvider):
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, [greet])

        assert "tools" in payload
        decls = payload["tools"]
        assert len(decls) == 1
        assert decls[0]["type"] == "function"
        assert decls[0]["function"]["name"] == "greet"
        assert decls[0]["function"]["description"] == "Greet someone."
        assert "parameters" in decls[0]["function"]

    def test_no_tools_omits_key(self, provider: OllamaProvider):
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, None)
        assert "tools" not in payload

    def test_output_schema_sets_response_format(self, provider: OllamaProvider):
        messages = [Message(role="user", content="hi")]
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        payload = provider._build_payload(messages, None, output_schema=schema)

        assert "response_format" in payload
        rf = payload["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "response"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"]["type"] == "object"
        assert "city" in rf["json_schema"]["schema"]["properties"]

    def test_no_output_schema_omits_response_format(self, provider: OllamaProvider):
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, None)
        assert "response_format" not in payload

    def test_output_schema_with_tools_both_present(self, provider: OllamaProvider):
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        messages = [Message(role="user", content="hi")]
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        payload = provider._build_payload(messages, [greet], output_schema=schema)

        assert "tools" in payload
        assert "response_format" in payload

    def test_full_conversation_round_trip(self, provider: OllamaProvider):
        messages = [
            Message(role="system", content="You are a helper."),
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(name="add", arguments={"a": 2, "b": 2}, id="c1")],
            ),
            Message(
                role="tool",
                tool_result=ToolResult(content="4", tool_call_id="c1", tool_name="add"),
            ),
        ]
        payload = provider._build_payload(messages, None)

        assert len(payload["messages"]) == 4
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][2]["role"] == "assistant"
        assert payload["messages"][3]["role"] == "tool"
        assert payload["messages"][3]["tool_call_id"] == "c1"


# ── response parsing ────────────────────────────────────────────────────────


class TestParseResponse:
    def test_text_response(self, provider: OllamaProvider):
        data = _ok_response("Hello!")
        msg = provider._parse_response(data)

        assert msg.role == "assistant"
        assert msg.content == "Hello!"
        assert msg.tool_calls is None

    def test_function_call_response(self, provider: OllamaProvider):
        data = _tool_call_response(
            name="get_weather",
            arguments={"city": "Tokyo"},
            call_id="fc_1",
        )
        msg = provider._parse_response(data)

        assert msg.role == "assistant"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "get_weather"
        assert msg.tool_calls[0].arguments == {"city": "Tokyo"}
        assert msg.tool_calls[0].id == "fc_1"
        assert msg.tool_calls[0].thought_signature is None

    def test_multiple_function_calls(self, provider: OllamaProvider):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "a", "arguments": "{}"},
                        },
                        {
                            "id": "2",
                            "type": "function",
                            "function": {"name": "b", "arguments": '{"x": 1}'},
                        },
                    ],
                }
            }]
        }
        msg = provider._parse_response(data)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].name == "a"
        assert msg.tool_calls[1].arguments == {"x": 1}

    def test_content_and_tool_calls_both_present(self, provider: OllamaProvider):
        data = _tool_call_response(
            name="search", arguments={"q": "test"}, content="Searching..."
        )
        msg = provider._parse_response(data)

        assert msg.content == "Searching..."
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].name == "search"

    def test_missing_tool_calls_key(self, provider: OllamaProvider):
        data = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        msg = provider._parse_response(data)

        assert msg.content == "hi"
        assert msg.tool_calls is None

    def test_tool_calls_null(self, provider: OllamaProvider):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "tool_calls": None,
                }
            }]
        }
        msg = provider._parse_response(data)

        assert msg.content == "hi"
        assert msg.tool_calls is None

    def test_empty_content_becomes_none(self, provider: OllamaProvider):
        data = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        msg = provider._parse_response(data)
        assert msg.content is None

    def test_invalid_json_arguments_raises(self, provider: OllamaProvider):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "bad",
                            "arguments": "{not valid json",
                        },
                    }],
                }
            }]
        }
        with pytest.raises(RuntimeError, match="invalid JSON in tool call"):
            provider._parse_response(data)

    def test_arguments_already_dict(self, provider: OllamaProvider):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "fn",
                            "arguments": {"already": "dict"},
                        },
                    }],
                }
            }]
        }
        msg = provider._parse_response(data)
        assert msg.tool_calls[0].arguments == {"already": "dict"}


# ── chat (HTTP integration) ─────────────────────────────────────────────────


class TestChat:
    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_happy_path(self, mock_urlopen, provider: OllamaProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response("hello"))

        msg = provider.chat([Message(role="user", content="hi")])
        assert msg.content == "hello"
        assert msg.role == "assistant"

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_correct_url(self, mock_urlopen, provider: OllamaProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:11434/v1/chat/completions"

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_no_auth_header(self, mock_urlopen, provider: OllamaProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-goog-api-key") is None
        assert req.get_header("Authorization") is None

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_content_type_header(self, mock_urlopen, provider: OllamaProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_timeout_passed(self, mock_urlopen, provider: OllamaProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 120

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_custom_timeout(self, mock_urlopen):
        provider = OllamaProvider(model="llama3.2", timeout=30)
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 30

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_url_error_gives_helpful_message(self, mock_urlopen):
        provider = OllamaProvider(model="llama3.2")
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            provider.chat([Message(role="user", content="test")])

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_404_gives_pull_hint(self, mock_urlopen):
        provider = OllamaProvider(model="nonexistent-model")
        fp = MagicMock()
        fp.read.return_value = b'{"error": "model not found"}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://test", code=404, msg="Not Found", hdrs={}, fp=fp,
        )

        with pytest.raises(RuntimeError, match="not found.*ollama pull"):
            provider.chat([Message(role="user", content="test")])

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_other_http_error(self, mock_urlopen, provider: OllamaProvider):
        fp = MagicMock()
        fp.read.return_value = b'{"error": "server error"}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://test", code=500, msg="Internal Server Error",
            hdrs={}, fp=fp,
        )

        with pytest.raises(RuntimeError, match="500"):
            provider.chat([Message(role="user", content="test")])

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_custom_base_url(self, mock_urlopen):
        provider = OllamaProvider(
            model="llama3.2", base_url="http://myhost:9999"
        )
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://myhost:9999/v1/chat/completions"


# ── auto config ─────────────────────────────────────────────────────────────


class TestAutoConfig:
    def test_default_base_url(self):
        provider = OllamaProvider(model="llama3.2")
        assert provider._base_url == "http://localhost:11434"

    def test_default_model(self):
        provider = OllamaProvider()
        assert provider.model == "llama3.2"

    @patch.dict("os.environ", {"OLLAMA_HOST": "http://custom:5555/"}, clear=False)
    def test_base_url_from_env(self):
        provider = OllamaProvider()
        assert provider._base_url == "http://custom:5555"

    def test_explicit_base_url_beats_env(self):
        provider = OllamaProvider(base_url="http://explicit:1234")
        assert provider._base_url == "http://explicit:1234"

    @patch.dict("os.environ", {"OLLAMA_MODEL": "mistral"}, clear=False)
    def test_model_from_env(self):
        provider = OllamaProvider()
        assert provider.model == "mistral"

    @patch.dict("os.environ", {"OLLAMA_MODEL": "mistral"}, clear=False)
    def test_explicit_model_beats_env(self):
        provider = OllamaProvider(model="codellama")
        assert provider.model == "codellama"

    def test_default_timeout(self):
        provider = OllamaProvider()
        assert provider.timeout == 120

    def test_custom_timeout(self):
        provider = OllamaProvider(timeout=30)
        assert provider.timeout == 30


# ── repr ────────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_model_and_url(self):
        provider = OllamaProvider(model="llama3.2", base_url="http://host:1234")
        r = repr(provider)
        assert "llama3.2" in r
        assert "http://host:1234" in r
        assert "OllamaProvider" in r


# ── arguments round-trip ────────────────────────────────────────────────────


class TestArgumentsRoundTrip:
    def test_dict_serialized_to_json_string_in_payload(self, provider: OllamaProvider):
        tc = ToolCall(name="fn", arguments={"key": "value", "n": 42}, id="c1")
        messages = [Message(role="assistant", tool_calls=[tc])]
        payload = provider._build_payload(messages, None)

        raw = payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(raw, str)
        assert json.loads(raw) == {"key": "value", "n": 42}

    def test_json_string_parsed_back_to_dict_in_response(self, provider: OllamaProvider):
        data = _tool_call_response(
            name="fn", arguments={"key": "value", "n": 42}
        )
        msg = provider._parse_response(data)

        assert msg.tool_calls[0].arguments == {"key": "value", "n": 42}
        assert isinstance(msg.tool_calls[0].arguments, dict)


# ── two providers do not interfere ──────────────────────────────────────────


class TestProviderIsolation:
    def test_different_configs_independent(self):
        p1 = OllamaProvider(model="llama3.2", base_url="http://host1:1111")
        p2 = OllamaProvider(model="mistral", base_url="http://host2:2222")

        assert p1.model == "llama3.2"
        assert p2.model == "mistral"
        assert p1._base_url == "http://host1:1111"
        assert p2._base_url == "http://host2:2222"

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_different_providers_hit_different_urls(self, mock_urlopen):
        p1 = OllamaProvider(model="llama3.2", base_url="http://host1:1111")
        p2 = OllamaProvider(model="mistral", base_url="http://host2:2222")
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        p1.chat([Message(role="user", content="test")])
        req1_url = mock_urlopen.call_args[0][0].full_url

        p2.chat([Message(role="user", content="test")])
        req2_url = mock_urlopen.call_args[0][0].full_url

        assert "host1:1111" in req1_url
        assert "host2:2222" in req2_url

    @patch("edge_agent.providers.ollama.urllib.request.urlopen")
    def test_different_providers_send_different_models(self, mock_urlopen):
        p1 = OllamaProvider(model="llama3.2")
        p2 = OllamaProvider(model="mistral")
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        p1.chat([Message(role="user", content="test")])
        body1 = json.loads(mock_urlopen.call_args[0][0].data)

        p2.chat([Message(role="user", content="test")])
        body2 = json.loads(mock_urlopen.call_args[0][0].data)

        assert body1["model"] == "llama3.2"
        assert body2["model"] == "mistral"
