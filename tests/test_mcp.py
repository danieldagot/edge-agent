from __future__ import annotations

import json
import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from edge_agent import Agent, MCPServer, load_mcp_config, tool
from edge_agent.mcp import _PROTOCOL_VERSION, _CLIENT_INFO
from edge_agent.types import Message, ToolCall

from tests.conftest import MockProvider


# ── helpers ──────────────────────────────────────────────────────────────────


def _jsonl(*messages: dict) -> bytes:
    """Encode a sequence of JSON-RPC messages as newline-delimited JSON."""
    return b"".join(json.dumps(m).encode() + b"\n" for m in messages)


def _ok(req_id: int, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: int, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


_INIT_RESULT = {
    "protocolVersion": _PROTOCOL_VERSION,
    "capabilities": {},
    "serverInfo": {"name": "test-server", "version": "1.0.0"},
}

_TOOLS_RESULT = {
    "tools": [
        {
            "name": "search",
            "description": "Search for information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a file from disk",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    ],
}


def _make_mock_process(stdout_bytes: bytes) -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = io.BytesIO(stdout_bytes)
    proc.stderr = io.BytesIO(b"")
    proc.wait = MagicMock(return_value=0)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    return proc


# ── construction ─────────────────────────────────────────────────────────────


class TestMCPServerConstruction:
    def test_stores_name_and_command(self):
        server = MCPServer("test", command=["echo", "hello"])
        assert server.name == "test"
        assert server._command == ["echo", "hello"]

    def test_starts_disconnected(self):
        server = MCPServer("test", command=["echo"])
        assert not server.connected
        assert server.tools == []

    def test_repr_disconnected(self):
        server = MCPServer("test", command=["echo"])
        assert "disconnected" in repr(server)

    def test_env_stored(self):
        server = MCPServer("test", command=["echo"], env={"FOO": "bar"})
        assert server._env == {"FOO": "bar"}


# ── connect / handshake ─────────────────────────────────────────────────────


class TestMCPServerConnect:
    @patch("edge_agent.mcp.subprocess.Popen")
    def test_sends_initialize_and_tools_list(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        assert server.connected
        assert len(server.tools) == 2

        written_calls = server._process.stdin.write.call_args_list
        assert len(written_calls) == 3  # initialize, initialized notif, tools/list

        init_msg = json.loads(written_calls[0].args[0])
        assert init_msg["method"] == "initialize"
        assert init_msg["params"]["protocolVersion"] == _PROTOCOL_VERSION
        assert init_msg["params"]["clientInfo"] == _CLIENT_INFO
        assert "id" in init_msg

        notif_msg = json.loads(written_calls[1].args[0])
        assert notif_msg["method"] == "notifications/initialized"
        assert "id" not in notif_msg

        tools_msg = json.loads(written_calls[2].args[0])
        assert tools_msg["method"] == "tools/list"
        assert "id" in tools_msg

        server.close()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_connect_is_idempotent(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()
        server.connect()  # should be a no-op

        assert mock_popen.call_count == 1
        server.close()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_repr_connected(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        r = repr(server)
        assert "connected" in r
        assert "tools=2" in r
        server.close()


# ── tool discovery ───────────────────────────────────────────────────────────


class TestMCPToolDiscovery:
    @patch("edge_agent.mcp.subprocess.Popen")
    def test_tools_have_correct_metadata(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        search_tool = server.tools[0]
        assert search_tool.name == "search"
        assert search_tool.description == "Search for information"
        assert search_tool.parameters == _TOOLS_RESULT["tools"][0]["inputSchema"]

        read_tool = server.tools[1]
        assert read_tool.name == "read_file"
        assert read_tool.description == "Read a file from disk"

        server.close()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_empty_tools_list(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, {"tools": []}),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        assert server.tools == []
        server.close()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_tool_without_description(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, {"tools": [{"name": "bare", "inputSchema": {"type": "object", "properties": {}}}]}),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        assert server.tools[0].description == ""
        server.close()


# ── tool calling ─────────────────────────────────────────────────────────────


class TestMCPToolCalling:
    @patch("edge_agent.mcp.subprocess.Popen")
    def test_tool_call_sends_correct_request(self, mock_popen):
        call_result = {
            "content": [{"type": "text", "text": "result for: quantum"}],
            "isError": False,
        }
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
            _ok(3, call_result),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        search = server.tools[0]
        result = search(query="quantum")
        assert result == "result for: quantum"

        written = server._process.stdin.write.call_args_list
        call_msg = json.loads(written[-1].args[0])
        assert call_msg["method"] == "tools/call"
        assert call_msg["params"]["name"] == "search"
        assert call_msg["params"]["arguments"] == {"query": "quantum"}

        server.close()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_tool_call_multiple_text_parts(self, mock_popen):
        call_result = {
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ],
            "isError": False,
        }
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
            _ok(3, call_result),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        result = server.tools[0](query="test")
        assert result == "line one\nline two"

        server.close()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_tool_call_error_flag(self, mock_popen):
        call_result = {
            "content": [{"type": "text", "text": "file not found"}],
            "isError": True,
        }
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
            _ok(3, call_result),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        result = server.tools[1](path="/nonexistent")
        assert result == "file not found"

        server.close()


# ── error handling ───────────────────────────────────────────────────────────


class TestMCPErrorHandling:
    @patch("edge_agent.mcp.subprocess.Popen")
    def test_jsonrpc_error_raises_runtime_error(self, mock_popen):
        stdout = _jsonl(
            _error(1, -32600, "Invalid request"),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        with pytest.raises(RuntimeError, match="MCP error.*Invalid request"):
            server.connect()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_broken_pipe_raises_connection_error(self, mock_popen):
        proc = _make_mock_process(b"")
        proc.stdin.write.side_effect = BrokenPipeError("pipe closed")
        mock_popen.return_value = proc

        server = MCPServer("test", command=["test-server"])
        with pytest.raises(ConnectionError, match="not responding"):
            server.connect()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_empty_stdout_raises_connection_error(self, mock_popen):
        proc = _make_mock_process(b"")
        mock_popen.return_value = proc

        server = MCPServer("test", command=["test-server"])
        with pytest.raises(ConnectionError, match="closed the connection"):
            server.connect()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_invalid_json_raises_connection_error(self, mock_popen):
        proc = _make_mock_process(b"not json\n")
        mock_popen.return_value = proc

        server = MCPServer("test", command=["test-server"])
        with pytest.raises(ConnectionError, match="invalid JSON"):
            server.connect()


# ── context manager ──────────────────────────────────────────────────────────


class TestMCPContextManager:
    @patch("edge_agent.mcp.subprocess.Popen")
    def test_enter_connects_exit_closes(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        with server:
            assert server.connected
            assert len(server.tools) == 2

        assert not server.connected
        assert server.tools == []

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_close_terminates_process(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        proc = _make_mock_process(stdout)
        mock_popen.return_value = proc

        server = MCPServer("test", command=["test-server"])
        server.connect()
        server.close()

        proc.terminate.assert_called_once()

    def test_close_without_connect_is_safe(self):
        server = MCPServer("test", command=["echo"])
        server.close()  # should not raise


# ── skipping notifications ───────────────────────────────────────────────────


class TestMCPNotificationSkipping:
    @patch("edge_agent.mcp.subprocess.Popen")
    def test_skips_server_notifications(self, mock_popen):
        """Notifications (messages without an ``id``) between the request
        and response should be silently skipped."""
        notification = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 50}}
        stdout = _jsonl(
            notification,
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        server.connect()

        assert server.connected
        assert len(server.tools) == 2
        server.close()


# ── agent integration ────────────────────────────────────────────────────────


class TestMCPAgentIntegration:
    @patch("edge_agent.mcp.subprocess.Popen")
    def test_agent_merges_mcp_tools(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        @tool
        def local_tool(x: str) -> str:
            """A local tool."""
            return x

        server = MCPServer("test", command=["test-server"])
        provider = MockProvider([Message(role="assistant", content="ok")])

        agent = Agent(
            name="mcp-test",
            instructions="Test.",
            provider=provider,
            tools=[local_tool],
            mcp_servers=[server],
        )

        assert "local_tool" in agent.tools
        assert "search" in agent.tools
        assert "read_file" in agent.tools
        assert len(agent.tools) == 3

        agent.close()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_agent_auto_connects_server(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        assert not server.connected

        provider = MockProvider([Message(role="assistant", content="ok")])
        agent = Agent(
            name="mcp-test",
            instructions="Test.",
            provider=provider,
            mcp_servers=[server],
        )

        assert server.connected
        agent.close()

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_duplicate_mcp_tool_name_raises(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, {"tools": [{"name": "search", "inputSchema": {"type": "object", "properties": {}}}]}),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        @tool
        def search(query: str) -> str:
            """Local search."""
            return query

        server = MCPServer("test", command=["test-server"])
        provider = MockProvider([Message(role="assistant", content="ok")])

        with pytest.raises(ValueError, match="Duplicate tool name.*search"):
            Agent(
                name="test",
                instructions="Test.",
                provider=provider,
                tools=[search],
                mcp_servers=[server],
            )

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_agent_close_closes_mcp_servers(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        proc = _make_mock_process(stdout)
        mock_popen.return_value = proc

        server = MCPServer("test", command=["test-server"])
        provider = MockProvider([Message(role="assistant", content="ok")])

        agent = Agent(
            name="mcp-test",
            instructions="Test.",
            provider=provider,
            mcp_servers=[server],
        )

        agent.close()
        proc.terminate.assert_called_once()
        assert not server.connected

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_agent_context_manager(self, mock_popen):
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])
        provider = MockProvider([Message(role="assistant", content="ok")])

        with Agent(
            name="mcp-test",
            instructions="Test.",
            provider=provider,
            mcp_servers=[server],
        ) as agent:
            assert "search" in agent.tools

        assert not server.connected

    @patch("edge_agent.mcp.subprocess.Popen")
    def test_agent_calls_mcp_tool_via_loop(self, mock_popen):
        """The agent loop should proxy MCP tool calls correctly."""
        call_result = {
            "content": [{"type": "text", "text": "search results here"}],
            "isError": False,
        }
        stdout = _jsonl(
            _ok(1, _INIT_RESULT),
            _ok(2, _TOOLS_RESULT),
            _ok(3, call_result),
        )
        mock_popen.return_value = _make_mock_process(stdout)

        server = MCPServer("test", command=["test-server"])

        responses = [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(name="search", arguments={"query": "hello"}, id="c1"),
                ],
            ),
            Message(role="assistant", content="Found: search results here"),
        ]
        provider = MockProvider(responses)

        with Agent(
            name="mcp-test",
            instructions="Use tools.",
            provider=provider,
            mcp_servers=[server],
        ) as agent:
            result = agent.run("Search for hello")

        assert result.output == "Found: search results here"

        tool_result_msg = provider.call_log[1][0][-1]
        assert tool_result_msg.role == "tool"
        assert tool_result_msg.tool_result.content == "search results here"


# ── agent without mcp (regression) ──────────────────────────────────────────


class TestAgentWithoutMCP:
    def test_agent_works_without_mcp_servers(self):
        """Adding the mcp_servers param must not break existing agents."""
        provider = MockProvider([Message(role="assistant", content="hello")])
        agent = Agent(
            name="no-mcp",
            instructions="Test.",
            provider=provider,
        )
        result = agent.run("hi")
        assert result.output == "hello"
        assert agent.mcp_servers == []

    def test_close_without_mcp_is_safe(self):
        provider = MockProvider([Message(role="assistant", content="ok")])
        agent = Agent(name="no-mcp", instructions="Test.", provider=provider)
        agent.close()  # should not raise

    def test_context_manager_without_mcp(self):
        provider = MockProvider([Message(role="assistant", content="ok")])
        with Agent(name="no-mcp", instructions="Test.", provider=provider) as agent:
            result = agent.run("hi")
        assert result.output == "ok"


# ── load_mcp_config ──────────────────────────────────────────────────────────


def _write_config(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data))


_SAMPLE_CONFIG = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {"HOME": "/tmp"},
        },
        "time": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-time"],
        },
        "bare": {
            "command": "python",
        },
    }
}


class TestLoadMcpConfig:
    def test_load_all_servers(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg)

        assert set(result) == {"filesystem", "time", "bare"}
        assert all(isinstance(v, MCPServer) for v in result.values())

    def test_server_names_match_keys(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg)

        for name, server in result.items():
            assert server.name == name

    def test_command_and_args_combined(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg)

        assert result["filesystem"]._command == [
            "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"
        ]
        assert result["time"]._command == [
            "npx", "-y", "@modelcontextprotocol/server-time"
        ]

    def test_args_optional_defaults_to_empty(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg)

        assert result["bare"]._command == ["python"]

    def test_env_is_passed_through(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg)

        assert result["filesystem"]._env == {"HOME": "/tmp"}

    def test_env_optional_defaults_to_none(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg)

        assert result["time"]._env is None
        assert result["bare"]._env is None

    def test_load_subset_by_name(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg, servers=["filesystem"])

        assert list(result) == ["filesystem"]
        assert result["filesystem"]._command[0] == "npx"

    def test_load_multiple_specific_servers(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg, servers=["filesystem", "bare"])

        assert set(result) == {"filesystem", "bare"}

    def test_unknown_server_name_raises_value_error(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        with pytest.raises(ValueError, match="nonexistent"):
            load_mcp_config(cfg, servers=["nonexistent"])

    def test_error_message_lists_available_servers(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        with pytest.raises(ValueError, match="Available"):
            load_mcp_config(cfg, servers=["missing"])

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mcp_config(tmp_path / "does_not_exist.json")

    def test_invalid_json_raises(self, tmp_path):
        cfg = tmp_path / "bad.json"
        cfg.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_mcp_config(cfg)

    def test_missing_mcp_servers_key_raises(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({"servers": {}}))

        with pytest.raises(KeyError):
            load_mcp_config(cfg)

    def test_accepts_string_path(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(str(cfg))

        assert "filesystem" in result

    def test_servers_are_not_connected(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        _write_config(_SAMPLE_CONFIG, cfg)

        result = load_mcp_config(cfg)

        for server in result.values():
            assert not server.connected
