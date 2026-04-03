from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from edge_agent.providers.gemini import GeminiProvider
from edge_agent.tool import Tool, tool
from edge_agent.types import Message, ToolCall


@pytest.fixture
def provider() -> GeminiProvider:
    return GeminiProvider(api_key="test-key", model="gemini-2.0-flash")


def _mock_urlopen(response_body: dict):
    """Return a context-manager mock for urllib.request.urlopen."""
    body = json.dumps(response_body).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestBuildPayload:
    def test_system_message_becomes_system_instruction(self, provider: GeminiProvider):
        messages = [Message(role="system", content="Be helpful.")]
        payload = provider._build_payload(messages, None)

        assert "system_instruction" in payload
        assert payload["system_instruction"]["parts"] == [{"text": "Be helpful."}]
        assert payload["contents"] == []

    def test_user_message(self, provider: GeminiProvider):
        messages = [Message(role="user", content="Hello")]
        payload = provider._build_payload(messages, None)

        assert len(payload["contents"]) == 1
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][0]["parts"] == [{"text": "Hello"}]

    def test_assistant_text_message(self, provider: GeminiProvider):
        messages = [Message(role="assistant", content="Hi there")]
        payload = provider._build_payload(messages, None)

        assert payload["contents"][0]["role"] == "model"
        assert payload["contents"][0]["parts"] == [{"text": "Hi there"}]

    def test_assistant_tool_call_message(self, provider: GeminiProvider):
        tc = ToolCall(name="search", arguments={"q": "test"}, id="call_1")
        messages = [Message(role="assistant", tool_calls=[tc])]
        payload = provider._build_payload(messages, None)

        parts = payload["contents"][0]["parts"]
        assert len(parts) == 1
        assert parts[0]["functionCall"]["name"] == "search"
        assert parts[0]["functionCall"]["args"] == {"q": "test"}
        assert parts[0]["functionCall"]["id"] == "call_1"

    def test_tool_result_message(self, provider: GeminiProvider):
        from edge_agent.types import ToolResult

        tr = ToolResult(content="42", tool_call_id="call_1")
        messages = [Message(role="tool", tool_result=tr)]
        payload = provider._build_payload(messages, None)

        parts = payload["contents"][0]["parts"]
        assert "functionResponse" in parts[0]
        assert parts[0]["functionResponse"]["response"] == {"result": "42"}

    def test_tools_as_function_declarations(self, provider: GeminiProvider):
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, [greet])

        assert "tools" in payload
        decls = payload["tools"][0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "greet"
        assert decls[0]["description"] == "Greet someone."

    def test_no_tools_omits_key(self, provider: GeminiProvider):
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, None)
        assert "tools" not in payload


class TestParseResponse:
    def test_text_response(self, provider: GeminiProvider):
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello!"}],
                        "role": "model",
                    }
                }
            ]
        }
        msg = provider._parse_response(data)
        assert msg.role == "assistant"
        assert msg.content == "Hello!"
        assert msg.tool_calls is None

    def test_function_call_response(self, provider: GeminiProvider):
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "Tokyo"},
                                    "id": "fc_1",
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        msg = provider._parse_response(data)
        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "get_weather"
        assert msg.tool_calls[0].arguments == {"city": "Tokyo"}
        assert msg.tool_calls[0].id == "fc_1"

    def test_multiple_function_calls(self, provider: GeminiProvider):
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "tool_a",
                                    "args": {},
                                    "id": "1",
                                }
                            },
                            {
                                "functionCall": {
                                    "name": "tool_b",
                                    "args": {"x": 1},
                                    "id": "2",
                                }
                            },
                        ],
                        "role": "model",
                    }
                }
            ]
        }
        msg = provider._parse_response(data)
        assert len(msg.tool_calls) == 2


class TestChat:
    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_sends_correct_headers(self, mock_urlopen, provider: GeminiProvider):
        response_data = {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}], "role": "model"}}
            ]
        }
        mock_urlopen.return_value = _mock_urlopen(response_data)

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"
        assert req.get_header("X-goog-api-key") == "test-key"

    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_uses_correct_url(self, mock_urlopen, provider: GeminiProvider):
        response_data = {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}], "role": "model"}}
            ]
        }
        mock_urlopen.return_value = _mock_urlopen(response_data)

        provider.chat([Message(role="user", content="test")])

        req = mock_urlopen.call_args[0][0]
        assert "gemini-2.0-flash:generateContent" in req.full_url

    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_http_error_raises_runtime_error(self, mock_urlopen, provider: GeminiProvider):
        import urllib.error

        error_resp = MagicMock()
        error_resp.read.return_value = b'{"error": "bad request"}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://test",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=error_resp,
        )

        with pytest.raises(RuntimeError, match="400"):
            provider.chat([Message(role="user", content="test")])

    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_request_uses_timeout(self, mock_urlopen, provider: GeminiProvider):
        response_data = {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}], "role": "model"}}
            ]
        }
        mock_urlopen.return_value = _mock_urlopen(response_data)

        provider.chat([Message(role="user", content="test")])

        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 60


class TestRetryOn429:
    def _make_429_error(self, body: str = "") -> urllib.error.HTTPError:
        import urllib.error

        fp = MagicMock()
        fp.read.return_value = body.encode()
        return urllib.error.HTTPError(
            url="http://test", code=429, msg="Too Many Requests",
            hdrs={}, fp=fp,
        )

    def _ok_response(self) -> dict:
        return {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}], "role": "model"}}
            ]
        }

    @patch("edge_agent.providers.gemini.time.sleep")
    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_retries_on_429_then_succeeds(self, mock_urlopen, mock_sleep):
        provider = GeminiProvider(
            api_key="test-key", model="gemini-2.0-flash",
            max_retries=2, retry_backoff=1.0,
        )
        mock_urlopen.side_effect = [
            self._make_429_error(),
            _mock_urlopen(self._ok_response()),
        ]

        msg = provider.chat([Message(role="user", content="test")])
        assert msg.content == "ok"
        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once()

    @patch("edge_agent.providers.gemini.time.sleep")
    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_raises_after_max_retries_exhausted(self, mock_urlopen, mock_sleep):
        provider = GeminiProvider(
            api_key="test-key", model="gemini-2.0-flash",
            max_retries=2, retry_backoff=1.0,
        )
        mock_urlopen.side_effect = [
            self._make_429_error(),
            self._make_429_error(),
            self._make_429_error(),
        ]

        with pytest.raises(RuntimeError, match="after 2 retries"):
            provider.chat([Message(role="user", content="test")])
        assert mock_sleep.call_count == 2

    @patch("edge_agent.providers.gemini.time.sleep")
    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_uses_server_retry_delay(self, mock_urlopen, mock_sleep):
        provider = GeminiProvider(
            api_key="test-key", model="gemini-2.0-flash",
            max_retries=2, retry_backoff=1.0,
        )
        body = (
            '{"error":{"message":"Quota exceeded. '
            'Please retry in 12.5s."}}'
        )
        mock_urlopen.side_effect = [
            self._make_429_error(body),
            _mock_urlopen(self._ok_response()),
        ]

        provider.chat([Message(role="user", content="test")])
        mock_sleep.assert_called_once_with(12.5)

    @patch("edge_agent.providers.gemini.time.sleep")
    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_falls_back_to_backoff_when_no_delay_in_body(
        self, mock_urlopen, mock_sleep,
    ):
        provider = GeminiProvider(
            api_key="test-key", model="gemini-2.0-flash",
            max_retries=3, retry_backoff=5.0,
        )
        mock_urlopen.side_effect = [
            self._make_429_error('{"error":{}}'),
            self._make_429_error('{"error":{}}'),
            _mock_urlopen(self._ok_response()),
        ]

        provider.chat([Message(role="user", content="test")])
        assert mock_sleep.call_args_list[0][0] == (5.0,)   # backoff * 1
        assert mock_sleep.call_args_list[1][0] == (10.0,)   # backoff * 2

    @patch("edge_agent.providers.gemini.time.sleep")
    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_non_429_error_is_not_retried(self, mock_urlopen, mock_sleep):
        import urllib.error

        provider = GeminiProvider(
            api_key="test-key", model="gemini-2.0-flash",
            max_retries=3,
        )
        fp = MagicMock()
        fp.read.return_value = b'{"error":"bad"}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://test", code=400, msg="Bad Request",
            hdrs={}, fp=fp,
        )

        with pytest.raises(RuntimeError, match="400"):
            provider.chat([Message(role="user", content="test")])
        assert mock_urlopen.call_count == 1
        mock_sleep.assert_not_called()

    def test_parse_retry_delay_extracts_seconds(self):
        provider = GeminiProvider(api_key="k")
        body = "Please retry in 12.448987315s."
        assert provider._parse_retry_delay(body) == pytest.approx(12.448987315)

    def test_parse_retry_delay_returns_none_when_missing(self):
        provider = GeminiProvider(api_key="k")
        assert provider._parse_retry_delay("no delay here") is None

    @patch("edge_agent.providers.gemini.time.sleep")
    @patch("edge_agent.providers.gemini.urllib.request.urlopen")
    def test_no_retries_when_max_retries_is_zero(
        self, mock_urlopen, mock_sleep,
    ):
        provider = GeminiProvider(
            api_key="test-key", model="gemini-2.0-flash",
            max_retries=0,
        )
        mock_urlopen.side_effect = self._make_429_error()

        with pytest.raises(RuntimeError, match="429"):
            provider.chat([Message(role="user", content="test")])
        mock_sleep.assert_not_called()


class TestAutoConfig:
    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-key-123"}, clear=False)
    def test_resolves_api_key_from_gemini_env_var(self):
        provider = GeminiProvider()
        assert provider._api_key == "env-key-123"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key-456"}, clear=False)
    def test_resolves_api_key_from_google_env_var(self):
        provider = GeminiProvider()
        assert provider._api_key == "google-key-456"

    @patch.dict(
        "os.environ",
        {"GEMINI_API_KEY": "gemini-wins", "GOOGLE_API_KEY": "google-loses"},
        clear=False,
    )
    def test_gemini_key_takes_precedence_over_google_key(self):
        provider = GeminiProvider()
        assert provider._api_key == "gemini-wins"

    @patch.dict("os.environ", {}, clear=True)
    @patch("edge_agent.providers.gemini.load_dotenv")
    def test_raises_when_no_api_key_found(self, _mock_dotenv):
        with pytest.raises(EnvironmentError, match="No API key found"):
            GeminiProvider()

    def test_explicit_api_key_skips_env_lookup(self):
        provider = GeminiProvider(api_key="explicit-key")
        assert provider._api_key == "explicit-key"

    @patch.dict("os.environ", {"GEMINI_API_KEY": "k"}, clear=False)
    def test_default_model_used_when_no_env_override(self):
        provider = GeminiProvider()
        assert provider.model == GeminiProvider.DEFAULT_MODEL

    @patch.dict(
        "os.environ",
        {"GEMINI_API_KEY": "k", "TINYAGENT_MODEL": "gemini-2.0-flash"},
        clear=False,
    )
    def test_model_override_from_legacy_env_var(self):
        provider = GeminiProvider()
        assert provider.model == "gemini-2.0-flash"

    @patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "k",
            "EDGE_AGENT_MODEL": "gemini-edge",
            "TINYAGENT_MODEL": "gemini-legacy",
        },
        clear=False,
    )
    def test_edge_agent_model_env_beats_legacy_tinyagent_model(self):
        provider = GeminiProvider()
        assert provider.model == "gemini-edge"

    @patch.dict(
        "os.environ",
        {"GEMINI_API_KEY": "k", "TINYAGENT_MODEL": "gemini-2.0-flash"},
        clear=False,
    )
    def test_explicit_model_beats_env_var(self):
        provider = GeminiProvider(model="gemini-pro")
        assert provider.model == "gemini-pro"


class TestApiKeySafety:
    def test_api_key_not_in_repr(self):
        provider = GeminiProvider(api_key="sk-very-secret-key-12345")
        r = repr(provider)
        assert "very-secret" not in r
        assert "***2345" in r

    def test_api_key_not_a_public_attribute(self):
        provider = GeminiProvider(api_key="sk-secret")
        assert not hasattr(provider, "api_key")
        assert hasattr(provider, "_api_key")


class TestStructuredOutput:
    def test_output_schema_injects_generation_config(self, provider: GeminiProvider):
        messages = [Message(role="user", content="hi")]
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temp": {"type": "number"},
            },
            "required": ["city", "temp"],
        }
        payload = provider._build_payload(messages, None, output_schema=schema)

        assert "generationConfig" in payload
        config = payload["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseSchema"]["type"] == "object"
        assert "city" in config["responseSchema"]["properties"]

    def test_no_output_schema_omits_generation_config(self, provider: GeminiProvider):
        messages = [Message(role="user", content="hi")]
        payload = provider._build_payload(messages, None)
        assert "generationConfig" not in payload

    def test_output_schema_with_tools_both_present(self, provider: GeminiProvider):
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
        assert "generationConfig" in payload
