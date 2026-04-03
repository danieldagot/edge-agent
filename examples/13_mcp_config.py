"""MCP config file loading — manage multiple servers from a JSON file.

Demonstrates:
  - Defining multiple MCP servers in a JSON config file
  - Loading ALL servers with load_mcp_config()
  - Loading only SPECIFIC servers by name
  - Passing a selected server to an Agent

The JSON format is compatible with Claude Desktop's MCP config::

    {
      "mcpServers": {
        "server-name": {
          "command": "npx",
          "args": ["-y", "some-mcp-server"],
          "env": { "API_KEY": "..." }   // optional
        }
      }
    }

Prerequisites:
  npm install -g @modelcontextprotocol/server-filesystem

Run:  uv run python examples/13_mcp_config.py
"""

import json
import tempfile
from pathlib import Path

from edge_agent import Agent, load_mcp_config

# ── 1. Write a temporary config file ─────────────────────────────────────────
# In a real project this would be a committed mcp.json (or claude_desktop_config.json).

config = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        },
        "time": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-time"],
        },
    }
}

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False
) as tmp:
    json.dump(config, tmp)
    config_path = Path(tmp.name)

print(f"Config written to: {config_path}\n")

# ── 2. Load ALL servers defined in the config ─────────────────────────────────

all_servers = load_mcp_config(config_path)
print(f"All servers loaded: {list(all_servers)}")
# → ['filesystem', 'time']

# ── 3. Load only a SPECIFIC server by name ───────────────────────────────────

selected = load_mcp_config(config_path, servers=["filesystem"])
print(f"Selected servers: {list(selected)}")
# → ['filesystem']

fs_server = selected["filesystem"]
print(f"\nServer object: {fs_server}")

# ── 4. Use the selected server with an Agent ──────────────────────────────────

with fs_server:
    print(f"\nConnected — {len(fs_server.tools)} tool(s) discovered:")
    for t in fs_server.tools:
        print(f"  - {t.name}: {t.description}")

    with Agent(
        instructions="You are a helpful file assistant.",
        mcp_servers=[fs_server],
    ) as agent:
        result = agent.run("List the files in /tmp and give me a short summary.")
        print(f"\nAgent: {result}")

# ── 5. Requesting an unknown server raises ValueError ─────────────────────────

try:
    load_mcp_config(config_path, servers=["nonexistent"])
except ValueError as exc:
    print(f"\nExpected error: {exc}")

config_path.unlink()
