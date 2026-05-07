"""Amazon Bedrock demo — API key authentication with the Converse API.

Demonstrates:
  - Creating a BedrockProvider with an API key and model selection
  - Tool calling via the Converse API
  - Structured output with `output_type`
  - Inference config (temperature, max tokens)

Setup:
  1. Open the Amazon Bedrock console → API keys → Generate a key
  2. Export the key:
       export AWS_BEARER_TOKEN_BEDROCK=<your-key>
  3. (Optional) Set a default model:
       export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
  4. (Optional) Set region (defaults to us-east-1):
       export AWS_DEFAULT_REGION=us-east-1

Run:
  uv run python examples/15_bedrock.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from edge_agent import Agent, tool
from edge_agent.providers import BedrockProvider


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


@tool
def add(a: int, b: int) -> str:
    """Add two numbers together and return the result."""
    return str(a + b)


@dataclass
class CityWeather:
    city: str
    temperature: str
    conditions: str
    summary: str


def run_plain_chat(provider: BedrockProvider) -> None:
    """Simple text generation — no tools, no structured output."""
    print("── Plain Chat ──────────────────────────────────────────")
    agent = Agent(
        name="bedrock-chat",
        instructions="You are a helpful, concise assistant.",
        provider=provider,
    )
    result = agent.run("Explain what Amazon Bedrock is in two sentences.")
    print(f"AGENT: {result}\n")


def run_tool_use(provider: BedrockProvider) -> None:
    """Tool calling through the Converse API."""
    print("── Tool Use ────────────────────────────────────────────")
    agent = Agent(
        name="bedrock-tools",
        instructions="Use tools when they help answer the request.",
        provider=provider,
        tools=[get_weather, add],
    )
    queries = [
        "What is the weather in Tokyo?",
        "What is 17 + 25?",
    ]
    for q in queries:
        print(f"USER: {q}")
        print(f"AGENT: {agent.run(q)}\n")


def run_structured_output(provider: BedrockProvider) -> None:
    """Structured output with a dataclass — JSON prompting fallback."""
    print("── Structured Output ───────────────────────────────────")
    agent = Agent(
        name="bedrock-structured",
        instructions=(
            "You are a weather assistant. Use the get_weather tool "
            "to look up real data, then return the result."
        ),
        provider=provider,
        tools=[get_weather],
        output_type=CityWeather,
    )
    result = agent.run("What's the weather in London?")
    print(f"Raw output: {result.output}")
    if result.parsed:
        w = result.parsed
        print(f"Parsed: city={w.city}, temp={w.temperature}, "
              f"conditions={w.conditions}")
    print()


if __name__ == "__main__":
    try:
        # ── Create the provider ───────────────────────────────────
        #
        # Option A: everything from environment variables
        #   export AWS_BEARER_TOKEN_BEDROCK=<your-key>
        #   export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
        #   export AWS_DEFAULT_REGION=us-east-1
        #   provider = BedrockProvider()
        #
        # Option B: pass values explicitly
        provider = BedrockProvider(
            api_key="your-api-key",                                  # or set AWS_BEARER_TOKEN_BEDROCK env var
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",   # or set BEDROCK_MODEL_ID env var
            region_name="us-east-1",                                 # or set AWS_DEFAULT_REGION env var
            inference_config={"maxTokens": 1024, "temperature": 0.7},
        )

        print(f"Using {provider!r}\n")

        run_plain_chat(provider)
        run_tool_use(provider)
        run_structured_output(provider)
    except EnvironmentError as exc:
        print(f"Setup error: {exc}")
        print("\nTo get started:")
        print("  1. Open the Bedrock console → API keys → Generate")
        print("  2. export AWS_BEARER_TOKEN_BEDROCK=<your-key>")
        print("  3. uv run python examples/15_bedrock.py")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Runtime error: {exc}")
        sys.exit(1)
