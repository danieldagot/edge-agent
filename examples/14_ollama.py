"""Ollama demo — explicit local provider with tool calling.

Demonstrates:
  - Explicit `OllamaProvider` usage
  - Local tool calling against an Ollama model
  - Env-based configuration via `OLLAMA_MODEL` / `OLLAMA_HOST`

Run:
  ollama pull llama3.2
  uv run python examples/14_ollama.py
"""

import sys

from edge_agent import Agent, Session, tool
from edge_agent.providers import OllamaProvider


@tool
def add(a: int, b: int) -> str:
    """Add two numbers together and return the result."""
    return str(a + b)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (fake data for demo purposes)."""
    fake = {
        "london": "14C, cloudy with light rain",
        "new york": "22C, sunny",
        "tokyo": "18C, partly cloudy",
        "sydney": "25C, clear skies",
    }
    return fake.get(city.lower(), f"No weather data available for {city}")


provider = OllamaProvider(model="gpt-oss:20b")

agent = Agent(
    name="ollama-demo",
    instructions=(
        "You are a concise local assistant running through Ollama. "
        "Use tools when they help answer the request."
    ),
    provider=provider,
    tools=[add, get_weather],
)


def run_batch() -> None:
    print(f"Using {provider!r}")

    queries = [
        "What is 7 + 12?",
        "What's the weather like in Tokyo?",
        "Add 5 and 6, then tell me the weather in London.",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"USER: {q}")
        print(f"AGENT: {agent.run(q)}")


if __name__ == "__main__":
    try:
        if "--live" in sys.argv:
            Session(agent).start()
        else:
            run_batch()
    except RuntimeError as exc:
        print(exc)
        print("Start Ollama locally and make sure the configured model is pulled.")
        print("Examples:")
        print("  ollama serve")
        print("  ollama pull llama3.2")
        raise SystemExit(1)
