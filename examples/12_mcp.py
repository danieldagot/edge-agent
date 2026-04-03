"""MCP (Model Context Protocol) — connect to external tool servers.

Demonstrates:
  - Connecting to an MCP server over stdio
  - Using MCP tools alongside local @tool-decorated functions
  - Context manager for automatic cleanup

Prerequisites:
  This example uses the official filesystem MCP server.  Install it with:
    npm install -g @modelcontextprotocol/server-filesystem

  Or run it without installing globally (npx fetches it on the fly):
    npx -y @modelcontextprotocol/server-filesystem /tmp

Run:  uv run python examples/12_mcp.py
"""

from edge_agent import Agent, MCPServer, tool


@tool
def summarize(text: str) -> str:
    """Summarize the given text in one sentence."""
    return f"Summary: {text[:80]}..."


# Connect to the filesystem MCP server.
# It exposes tools like read_file, write_file, list_directory, etc.
server = MCPServer(
    "filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

# The context manager connects on entry and cleans up on exit.
with server:
    print(f"Connected to {server.name}")
    print(f"Discovered {len(server.tools)} tools:")
    for t in server.tools:
        print(f"  - {t.name}: {t.description}")

    # Pass the MCP server to an agent — its tools are merged with local tools.
    with Agent(
        instructions=(
            "You are a helpful file assistant. "
            "Use the available tools to work with files."
        ),
        tools=[summarize],
        mcp_servers=[server],
    ) as agent:
        result = agent.run("List the files in /tmp")
        print(f"\nAgent: {result}")
