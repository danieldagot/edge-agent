"""Multi-tool demo — tools declared across separate files, two approaches.

Demonstrates:
  1. FLAT approach: single agent loaded with ALL 15 tools (every tool sent every API call)
  2. ROUTER approach: chain with specialist agents, each holding only its domain tools
     (each API call only sends the tools relevant to that specialist)

Logging is enabled so you can see exactly which tools are sent per agent per turn.

Run:  uv run python examples/multi_tool_demo/main.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from edge_agent import Agent, Chain, Router, RunResult, tool
from edge_agent.providers.gemini import GeminiProvider
from tools import ALL_TOOLS, TOOLS_BY_DOMAIN


# ── logging setup ────────────────────────────────────────────────────────────
# Patch the provider's chat method to log which tools are actually sent.

_original_chat = None


def _instrumented_chat(self, messages, tools=None, output_schema=None):
    """Wrapper that logs the tool names sent in each API call."""
    if tools:
        names = [t.name for t in tools]
        logging.getLogger("edge_agent").info(
            "  API call with %d tools: %s", len(names), names,
        )
    else:
        logging.getLogger("edge_agent").info("  API call with 0 tools")
    return _original_chat(self, messages, tools, output_schema)


def enable_tool_logging():
    global _original_chat
    _original_chat = GeminiProvider.chat
    GeminiProvider.chat = _instrumented_chat

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("  %(name)s | %(message)s"))
    logger = logging.getLogger("edge_agent")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


# ── approach 1: single agent, all tools ──────────────────────────────────────

def run_flat(query: str) -> RunResult:
    """One agent that has every tool from every domain."""
    agent = Agent(
        name="do-it-all",
        instructions=(
            "You are a personal assistant with access to math, weather, "
            "notes, and calendar tools. Use the appropriate tools to help "
            "the user. Be concise."
        ),
        tools=ALL_TOOLS,
    )
    return agent.run(query)


# ── approach 2: router chain, scoped tools ───────────────────────────────────

def build_router_chain() -> Chain:
    """Router dispatches to specialist agents, each with only its own tools."""
    router = Router(
        name="dispatcher",
        instructions=(
            "Route the user's request to the best specialist.\n"
            "Choose: math-agent for calculations, weather-agent for "
            "weather queries, notes-agent for note management, or "
            "calendar-agent for events and scheduling."
        ),
    )

    math_agent = Agent(
        name="math-agent",
        instructions="You are a math specialist. Use your tools to compute answers. Be concise.",
        tools=TOOLS_BY_DOMAIN["math"],
    )

    weather_agent = Agent(
        name="weather-agent",
        instructions="You are a weather specialist. Use your tools to look up weather. Be concise.",
        tools=TOOLS_BY_DOMAIN["weather"],
    )

    notes_agent = Agent(
        name="notes-agent",
        instructions="You are a notes manager. Use your tools to manage the user's notes. Be concise.",
        tools=TOOLS_BY_DOMAIN["notes"],
    )

    calendar_agent = Agent(
        name="calendar-agent",
        instructions="You are a calendar manager. Use your tools to manage events. Be concise.",
        tools=TOOLS_BY_DOMAIN["calendar"],
    )

    return Chain(agents=[router, math_agent, weather_agent, notes_agent, calendar_agent])


def run_router(chain: Chain, query: str) -> RunResult:
    """Chain routes to the right specialist — only that specialist's tools are sent."""
    return chain.run(query)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    enable_tool_logging()

    tool_count = len(ALL_TOOLS)
    tool_names = [t.name for t in ALL_TOOLS]
    print(f"\n{'═' * 70}")
    print(f"  Multi-tool demo — {tool_count} tools across 4 domains")
    print(f"  All tools: {tool_names}")
    print(f"{'═' * 70}")

    for domain, tools in TOOLS_BY_DOMAIN.items():
        print(f"  {domain:>10}: {[t.name for t in tools]}")

    queries = [
        "What is 42 * 17?",
        "What's the weather in Paris?",
        "Save a note titled 'groceries' with content 'milk, eggs, bread'",
        "What events do I have on 2026-03-31?",
    ]

    # ── Approach 1: flat ─────────────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  APPROACH 1: Single agent with ALL tools")
    print(f"  (every API call sends all {tool_count} tools)")
    print(f"{'━' * 70}")

    for q in queries:
        print(f"\n  USER: {q}")
        result = run_flat(q)
        print(f"  RESULT: {result}")

    # ── Approach 2: router chain ─────────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("  APPROACH 2: Router chain with specialist agents")
    print("  (each API call only sends the specialist's tools)")
    print(f"{'━' * 70}")

    chain = build_router_chain()

    for q in queries:
        print(f"\n  USER: {q}")
        result = run_router(chain, q)
        print(f"  RESULT: {result}")

    print(f"\n{'═' * 70}")
    print("  Summary:")
    print(f"    Flat:   every call sends {tool_count} tools → bloated context")
    print(f"    Router: each call sends only 1-4 tools → focused context")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
