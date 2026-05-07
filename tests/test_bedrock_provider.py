from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from edge_agent.providers.bedrock import BedrockProvider
from edge_agent.tool import Tool, tool
from edge_agent.types import Message, ToolCall, ToolResult


@pytest.fixture
def provider() -> BedrockProvider:
    return BedrockProvider(api_key="test-bearer-token", model_id="anthropic.claude-3-haiku-20240307-v1:0")


def _mock_urlopen(response_body: dict):
    """Return a context-manager mock for urllib.request.urlopen."""
    body = json.dumps(response_body).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _ok_response(text: str = "Hello!") -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


def _tool_use_response(
    name: str = "get_weather",
    tool_input: dict | None = None,
    tool_use_id: str = "tu_1",
    text: str | None = None,
) -> dict:
    content: list[dict] = []
    if text is not None:
        content.append({"text": text})
    content.append({
        "toolUse": {
            "toolUseId": tool_use_id,
            "name": name,
            "input": tool_input or {},
        }
    })
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content,
            }
        },
        "stopReason": "tool_use",
    }


# ── payload construction ────────────────────────────────────────────────────


class TestBuildPayload:
    def test_system_message_extracted(self, provider: BedrockProvider):
        messages = [Message(role="system", content="Be helpful.")]
        payload = provider._build_payload(messages, None)

        assert "system" in payload
        assert payload["system"] == [{"text": "Be helpful."}]
        assert payload["messages"] == []

    def test_multiple_system_messages(self, provider: BedrockProvider):
        messages = [
            Message(role="system", content="Rule 1."),
            Message(role="system", content="Rule 2."),
        ]
        payload = provider._build_payload(messages, None)

        assert len(payload["system"]) == 2
        assert payload["system"][0] == {"text": "Rule 1."}
        assert payload["system"][1] == {"text": "Rule 2."}

    def test_user_message(self, provider: BedrockProvider):
        messages = [Message(role="user", content="Hello")]
        payload = provider._build_payload(messages, None)

        assert len(payload["messages"]) == 1
        msg = payload["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == [{"text": "Hello"}]

    def test_assistant_text_message(self, provider: BedrockProvider):
        messages = [Message(role="assistant", content="Hi there")]
        payload = provider._build_payload(messages, None)

        msg = payload["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"text": "Hi there"}]

    def test_assistant_tool_call_message(self, provider: BedrockProvider):
        tc = ToolCall(name="search", arguments={"q": "test"}, id="tu_1")
        messages = [Message(role="assistant", tool_calls=[tc])]
        payload = provider._build_payload(messages, None)

        msg = payload["messages"][0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1
        tu = msg["content"][0]["toolUse"]
        assert tu["name"] == "search"
        assert tu["input"] == {"q": "test"}
        assert tu["toolUseId"] == "tu_1"

    def test_assistant_text_and_tool_call(self, provider: BedrockProvider):
        tc = ToolCall(name="search", arguments={}, id="tu_1")
        messages = [Message(role="assistant", content="Thinking...", tool_calls=[tc])]
        payload = provider._build_payload(messages, None)

        content = payload["messages"][0]["content"]
        assert len(content) == 2
        assert content[0] == {"text": "Thinking..."}
        assert "toolUse" in content[1]

    def test_tool_result_message(self, provider: BedrockProvider):
        tr = ToolResult(content="42", tool_call_id="tu_1", tool_name="calc")
        messages = [Message(role="tool", tool_result=tr)]
        payload = provider._build_payload(messages, None)

        msg = payload["messages"][0]
        assert msg["role"] == "user"
        block = msg["content"][0]["toolResult"]
        assert block["toolUseId"] == "tu_1"
        assert block["content"] == [{"text": "42"}]

    def test_consecutive_tool_results_merged(self, provider: BedrockProvider):
        tr1 = ToolResult(content="result1", tool_call_id="tu_1")
        tr2 = ToolResult(content="result2", tool_call_id="tu_2")
        messages = [
            Message(role="tool", tool_result=tr1),
            Message(role="tool", tool_result=tr2),
        ]
        payload = provider._build_payload(messages, None)

        assert len(payload["messages"]) == 1
        msg = payload["messages"][0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["toolResult"]["toolUseId"] == "tu_1"
        assert msg["content"][1]["toolResult"]["toolUseId"] == "tu_2"

    def test_tool_result_none_id_falls_back(self, provider: BedrockProvider):
        tr = ToolResult(content="ok", tool_call_id=None)
        messages = [Message(role="tool", tool_result=tr)]
        payload = provider._build_payload(messages, None)

        assert payload["messages"][0]["content"][0]["toolResult"]["toolUseId"] == ""

    def test_tools_as_tool_config(self, provider: BedrockProvider):
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, [greet])

        assert "toolConfig" in payload
        tools = payload["toolConfig"]["tools"]
        assert len(tools) == 1
        spec = tools[0]["toolSpec"]
        assert spec["name"] == "greet"
        assert spec["description"] == "Greet someone."
        assert "json" in spec["inputSchema"]

    def test_no_tools_omits_tool_config(self, provider: BedrockProvider):
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, None)
        assert "toolConfig" not in payload

    def test_model_id_in_payload(self, provider: BedrockProvider):
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, None)
        assert payload["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_inference_config_included(self):
        provider = BedrockProvider(
            api_key="k",
            inference_config={"maxTokens": 512, "temperature": 0.7},
        )
        payload = provider._build_payload(
            [Message(role="user", content="hi")], None,
        )
        assert payload["inferenceConfig"] == {"maxTokens": 512, "temperature": 0.7}

    def test_additional_model_request_fields(self):
        provider = BedrockProvider(
            api_key="k",
            additional_model_request_fields={"top_k": 50},
        )
        payload = provider._build_payload(
            [Message(role="user", content="hi")], None,
        )
        assert payload["additionalModelRequestFields"] == {"top_k": 50}

    def test_no_inference_config_omits_key(self, provider: BedrockProvider):
        payload = provider._build_payload(
            [Message(role="user", content="hi")], None,
        )
        assert "inferenceConfig" not in payload
        assert "additionalModelRequestFields" not in payload


class TestOutputSchema:
    def test_json_prompt_injected_when_not_native(self, provider: BedrockProvider):
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, None, output_schema=schema)

        assert "system" in payload
        system_text = payload["system"][-1]["text"]
        assert "JSON" in system_text
        assert '"city"' in system_text

    def test_no_output_schema_no_injection(self, provider: BedrockProvider):
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, None)
        assert "system" not in payload

    def test_output_schema_with_tools_both_present(self, provider: BedrockProvider):
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        schema: dict[str, object] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, [greet], output_schema=schema)

        assert "toolConfig" in payload
        assert "system" in payload
        assert "JSON" in payload["system"][-1]["text"]

    def test_native_structured_output_skips_prompt(self):
        provider = BedrockProvider(
            api_key="k", supports_structured_output=True,
        )
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        payload = provider._build_payload(
            [Message(role="user", content="hi")], None, output_schema=schema,
        )
        assert "system" not in payload


class TestCapabilityChecks:
    def test_tools_rejected_when_not_supported(self):
        provider = BedrockProvider(
            api_key="k", supports_tool_use=False,
        )

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        with pytest.raises(RuntimeError, match="Tool use is not supported"):
            provider._build_payload(
                [Message(role="user", content="hi")], [greet],
            )

    def test_tools_allowed_when_supported(self, provider: BedrockProvider):
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        payload = provider._build_payload(
            [Message(role="user", content="hi")], [greet],
        )
        assert "toolConfig" in payload


class TestFullConversation:
    def test_round_trip(self, provider: BedrockProvider):
        messages = [
            Message(role="system", content="You are a helper."),
            Message(role="user", content="What is 2+2?"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(name="add", arguments={"a": 2, "b": 2}, id="tu_1")],
            ),
            Message(
                role="tool",
                tool_result=ToolResult(content="4", tool_call_id="tu_1", tool_name="add"),
            ),
        ]
        payload = provider._build_payload(messages, None)

        assert payload["system"] == [{"text": "You are a helper."}]
        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][2]["role"] == "user"
        assert "toolResult" in payload["messages"][2]["content"][0]


# ── response parsing ────────────────────────────────────────────────────────


class TestParseResponse:
    def test_text_response(self, provider: BedrockProvider):
        data = _ok_response("Hello!")
        msg = provider._parse_response(data)

        assert msg.role == "assistant"
        assert msg.content == "Hello!"
        assert msg.tool_calls is None

    def test_tool_use_response(self, provider: BedrockProvider):
        data = _tool_use_response(
            name="get_weather",
            tool_input={"city": "Tokyo"},
            tool_use_id="tu_1",
        )
        msg = provider._parse_response(data)

        assert msg.role == "assistant"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "get_weather"
        assert msg.tool_calls[0].arguments == {"city": "Tokyo"}
        assert msg.tool_calls[0].id == "tu_1"

    def test_text_and_tool_use(self, provider: BedrockProvider):
        data = _tool_use_response(
            name="search", tool_input={"q": "test"}, text="Let me search...",
        )
        msg = provider._parse_response(data)

        assert msg.content == "Let me search..."
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].name == "search"

    def test_multiple_tool_uses(self, provider: BedrockProvider):
        data = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "1", "name": "a", "input": {}}},
                        {"toolUse": {"toolUseId": "2", "name": "b", "input": {"x": 1}}},
                    ],
                }
            },
            "stopReason": "tool_use",
        }
        msg = provider._parse_response(data)

        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].name == "a"
        assert msg.tool_calls[1].arguments == {"x": 1}

    def test_empty_content_list(self, provider: BedrockProvider):
        data = {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": "end_turn",
        }
        msg = provider._parse_response(data)
        assert msg.content is None
        assert msg.tool_calls is None

    def test_malformed_response_missing_output(self, provider: BedrockProvider):
        with pytest.raises(RuntimeError, match="malformed response"):
            provider._parse_response({})

    def test_malformed_response_missing_message(self, provider: BedrockProvider):
        with pytest.raises(RuntimeError, match="malformed response"):
            provider._parse_response({"output": {}})


# ── chat (HTTP integration) ─────────────────────────────────────────────────


class TestChat:
    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_happy_path(self, mock_urlopen, provider: BedrockProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response("hello"))

        msg = provider.chat([Message(role="user", content="hi")])
        assert msg.content == "hello"
        assert msg.role == "assistant"

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_correct_url(self, mock_urlopen, provider: BedrockProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert "bedrock-runtime.us-east-1.amazonaws.com" in req.full_url
        assert "anthropic.claude-3-haiku-20240307-v1:0" in req.full_url
        assert "/converse" in req.full_url

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_bearer_auth_header(self, mock_urlopen, provider: BedrockProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer test-bearer-token"

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_content_type_header(self, mock_urlopen, provider: BedrockProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_timeout_passed(self, mock_urlopen, provider: BedrockProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 120

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_custom_timeout(self, mock_urlopen):
        provider = BedrockProvider(api_key="k", timeout=30)
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 30

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_model_id_not_in_request_body(self, mock_urlopen, provider: BedrockProvider):
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert "modelId" not in body

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_custom_region(self, mock_urlopen):
        provider = BedrockProvider(
            api_key="k", region_name="eu-west-1",
        )
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert "bedrock-runtime.eu-west-1.amazonaws.com" in req.full_url


# ── error handling ──────────────────────────────────────────────────────────


class TestErrorHandling:
    def _make_http_error(self, code: int, body: str = "") -> urllib.error.HTTPError:
        fp = MagicMock()
        fp.read.return_value = body.encode()
        return urllib.error.HTTPError(
            url="http://test", code=code, msg="Error",
            hdrs={}, fp=fp,
        )

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_403_access_denied(self, mock_urlopen, provider: BedrockProvider):
        body = json.dumps({"message": "Access denied"})
        mock_urlopen.side_effect = self._make_http_error(403, body)

        with pytest.raises(RuntimeError, match="access denied"):
            provider.chat([Message(role="user", content="test")])

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_400_validation_error(self, mock_urlopen, provider: BedrockProvider):
        body = json.dumps({"message": "Model not found"})
        mock_urlopen.side_effect = self._make_http_error(400, body)

        with pytest.raises(RuntimeError, match="validation failed"):
            provider.chat([Message(role="user", content="test")])

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_404_model_not_found(self, mock_urlopen, provider: BedrockProvider):
        body = json.dumps({"message": "Not found"})
        mock_urlopen.side_effect = self._make_http_error(404, body)

        with pytest.raises(RuntimeError, match="not found"):
            provider.chat([Message(role="user", content="test")])

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_500_server_error(self, mock_urlopen, provider: BedrockProvider):
        body = json.dumps({"message": "Internal error"})
        mock_urlopen.side_effect = self._make_http_error(500, body)

        with pytest.raises(RuntimeError, match="500"):
            provider.chat([Message(role="user", content="test")])

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_url_error_connection_failure(self, mock_urlopen, provider: BedrockProvider):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(RuntimeError, match="Cannot connect to Bedrock"):
            provider.chat([Message(role="user", content="test")])


class TestRetryOn429:
    def _make_429_error(self, body: str = "") -> urllib.error.HTTPError:
        fp = MagicMock()
        fp.read.return_value = body.encode()
        return urllib.error.HTTPError(
            url="http://test", code=429, msg="Too Many Requests",
            hdrs={}, fp=fp,
        )

    @patch("edge_agent.providers.bedrock.time.sleep")
    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_retries_on_429_then_succeeds(self, mock_urlopen, mock_sleep):
        provider = BedrockProvider(
            api_key="k", max_retries=2, retry_backoff=1.0,
        )
        mock_urlopen.side_effect = [
            self._make_429_error(),
            _mock_urlopen(_ok_response()),
        ]

        msg = provider.chat([Message(role="user", content="test")])
        assert msg.content == "Hello!"
        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once()

    @patch("edge_agent.providers.bedrock.time.sleep")
    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_raises_after_max_retries_exhausted(self, mock_urlopen, mock_sleep):
        provider = BedrockProvider(
            api_key="k", max_retries=2, retry_backoff=1.0,
        )
        mock_urlopen.side_effect = [
            self._make_429_error(),
            self._make_429_error(),
            self._make_429_error(),
        ]

        with pytest.raises(RuntimeError, match="throttled after 2 retries"):
            provider.chat([Message(role="user", content="test")])
        assert mock_sleep.call_count == 2

    @patch("edge_agent.providers.bedrock.time.sleep")
    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_backoff_delay_increases(self, mock_urlopen, mock_sleep):
        provider = BedrockProvider(
            api_key="k", max_retries=3, retry_backoff=5.0,
        )
        mock_urlopen.side_effect = [
            self._make_429_error(),
            self._make_429_error(),
            _mock_urlopen(_ok_response()),
        ]

        provider.chat([Message(role="user", content="test")])
        assert mock_sleep.call_args_list[0][0] == (5.0,)   # backoff * 1
        assert mock_sleep.call_args_list[1][0] == (10.0,)  # backoff * 2

    @patch("edge_agent.providers.bedrock.time.sleep")
    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_non_429_error_is_not_retried(self, mock_urlopen, mock_sleep):
        provider = BedrockProvider(api_key="k", max_retries=3)
        body = json.dumps({"message": "bad"})
        fp = MagicMock()
        fp.read.return_value = body.encode()
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://test", code=400, msg="Bad Request",
            hdrs={}, fp=fp,
        )

        with pytest.raises(RuntimeError, match="validation failed"):
            provider.chat([Message(role="user", content="test")])
        assert mock_urlopen.call_count == 1
        mock_sleep.assert_not_called()

    @patch("edge_agent.providers.bedrock.time.sleep")
    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_no_retries_when_max_retries_is_zero(self, mock_urlopen, mock_sleep):
        provider = BedrockProvider(api_key="k", max_retries=0)
        mock_urlopen.side_effect = self._make_429_error()

        with pytest.raises(RuntimeError, match="429"):
            provider.chat([Message(role="user", content="test")])
        mock_sleep.assert_not_called()


# ── auto config ─────────────────────────────────────────────────────────────


class TestAutoConfig:
    def test_default_model(self):
        provider = BedrockProvider(api_key="k")
        assert provider.model_id == BedrockProvider.DEFAULT_MODEL

    def test_default_region(self):
        provider = BedrockProvider(api_key="k")
        assert provider.region_name == "us-east-1"

    @patch.dict("os.environ", {"BEDROCK_MODEL_ID": "amazon.titan-text-lite-v1"}, clear=False)
    def test_model_from_env(self):
        provider = BedrockProvider(api_key="k")
        assert provider.model_id == "amazon.titan-text-lite-v1"

    @patch.dict("os.environ", {"BEDROCK_MODEL_ID": "amazon.titan-text-lite-v1"}, clear=False)
    def test_explicit_model_beats_env(self):
        provider = BedrockProvider(api_key="k", model_id="anthropic.claude-3-haiku-20240307-v1:0")
        assert provider.model_id == "anthropic.claude-3-haiku-20240307-v1:0"

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "eu-west-1"}, clear=False)
    def test_region_from_env(self):
        provider = BedrockProvider(api_key="k")
        assert provider.region_name == "eu-west-1"

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "eu-west-1"}, clear=False)
    def test_explicit_region_beats_env(self):
        provider = BedrockProvider(api_key="k", region_name="us-west-2")
        assert provider.region_name == "us-west-2"

    @patch.dict("os.environ", {"AWS_BEARER_TOKEN_BEDROCK": "env-token-123"}, clear=False)
    def test_api_key_from_env(self):
        provider = BedrockProvider()
        assert provider._api_key == "env-token-123"

    @patch.dict("os.environ", {"AWS_BEARER_TOKEN_BEDROCK": "env-token"}, clear=False)
    def test_explicit_api_key_beats_env(self):
        provider = BedrockProvider(api_key="explicit-token")
        assert provider._api_key == "explicit-token"

    @patch.dict("os.environ", {}, clear=True)
    @patch("edge_agent.providers.bedrock.load_dotenv")
    def test_raises_when_no_api_key(self, _mock_dotenv):
        with pytest.raises(EnvironmentError, match="No Bedrock API key found"):
            BedrockProvider()

    def test_invalid_model_id_raises(self):
        with pytest.raises(ValueError, match="Invalid Bedrock model ID"):
            BedrockProvider(api_key="k", model_id="bad model id with spaces")

    def test_default_timeout(self):
        provider = BedrockProvider(api_key="k")
        assert provider.timeout == 120

    def test_custom_timeout(self):
        provider = BedrockProvider(api_key="k", timeout=30)
        assert provider.timeout == 30


# ── repr ────────────────────────────────────────────────────────────────────


class TestRepr:
    def test_api_key_masked_in_repr(self):
        provider = BedrockProvider(api_key="super-secret-bearer-token-12345")
        r = repr(provider)
        assert "super-secret" not in r
        assert "***2345" in r

    def test_repr_contains_model_and_region(self):
        provider = BedrockProvider(
            api_key="k",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name="eu-west-1",
        )
        r = repr(provider)
        assert "BedrockProvider" in r
        assert "anthropic.claude-3-haiku-20240307-v1:0" in r
        assert "eu-west-1" in r

    def test_api_key_not_public_attribute(self):
        provider = BedrockProvider(api_key="secret")
        assert not hasattr(provider, "api_key")
        assert hasattr(provider, "_api_key")


# ── provider isolation ──────────────────────────────────────────────────────


class TestProviderIsolation:
    def test_different_configs_independent(self):
        p1 = BedrockProvider(api_key="k1", model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-east-1")
        p2 = BedrockProvider(api_key="k2", model_id="amazon.titan-text-lite-v1", region_name="eu-west-1")

        assert p1.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
        assert p2.model_id == "amazon.titan-text-lite-v1"
        assert p1.region_name == "us-east-1"
        assert p2.region_name == "eu-west-1"

    @patch("edge_agent.providers.bedrock.urllib.request.urlopen")
    def test_different_providers_hit_different_urls(self, mock_urlopen):
        p1 = BedrockProvider(api_key="k", region_name="us-east-1")
        p2 = BedrockProvider(api_key="k", region_name="eu-west-1")
        mock_urlopen.return_value = _mock_urlopen(_ok_response())

        p1.chat([Message(role="user", content="test")])
        req1_url = mock_urlopen.call_args[0][0].full_url

        p2.chat([Message(role="user", content="test")])
        req2_url = mock_urlopen.call_args[0][0].full_url

        assert "us-east-1" in req1_url
        assert "eu-west-1" in req2_url
