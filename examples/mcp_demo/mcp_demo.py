"""MCP demo — load servers from mcp.json and query them via an agent.

This script reads MCP server definitions from mcp.json (Claude Desktop-compatible
format) and connects to one of the defined servers.

Prerequisites:
  Node.js must be installed so npx can fetch the MCP servers on first run.
  No other installation is required — npx downloads servers automatically.

Edit the constants below, then run:
  uv run python examples/mcp_demo/mcp_demo.py
"""

from pathlib import Path

from edge_agent import Agent, load_mcp_config

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG_FILE = Path(__file__).parent / "mcp.json"
SERVER_NAME = "filesystem"      # which server to load from the config
QUERY       = "List the files in /tmp and give me a short summary."

# ── Run ───────────────────────────────────────────────────────────────────────

servers = load_mcp_config(CONFIG_FILE, servers=[SERVER_NAME])
server  = servers[SERVER_NAME]

with server:
    print(f"Connected to '{server.name}'")
    print(f"Discovered {len(server.tools)} tool(s):")
    for t in server.tools:
        print(f"  - {t.name}: {t.description}")

    with Agent(
        instructions=(
            "You are a helpful assistant. "
            "Use tools to answer questions. Be concise."
        ),
        mcp_servers=[server],
    ) as agent:
        result = agent.run(QUERY)
        print(f"\nAgent: {result}")
