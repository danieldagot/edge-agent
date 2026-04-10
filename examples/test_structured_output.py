"""Diagnostic script for Ollama structured output.

Tests structured output in isolation — no tools, just schema enforcement.
Helps verify the response_format field works across models.

Run:
  uv run python examples/test_structured_output.py
"""

import json
import time
from dataclasses import dataclass

from edge_agent import Agent
from edge_agent.providers import OllamaProvider

MODELS = [
    "qwen3:4b-instruct",
    "gemma4:e4b",
    "llama3.1:8b",
]


@dataclass
class MovieReview:
    title: str
    year: int
    rating: float
    summary: str


@dataclass
class CityReport:
    city: str
    temperature: str
    conditions: str
    recommendation: str


TESTS = [
    {
        "label": "simple dataclass (no tools)",
        "question": "Write a short review of the movie Inception (2010).",
        "output_type": MovieReview,
        "tools": [],
        "check": lambda parsed: (
            isinstance(parsed, MovieReview)
            and parsed.title != ""
            and parsed.year > 0
            and parsed.rating > 0
        ),
    },
    {
        "label": "dataclass with tool",
        "question": (
            "Get the weather in Sydney and return a structured city report "
            "with a recommendation on what to wear."
        ),
        "output_type": CityReport,
        "tools": ["weather"],
        "check": lambda parsed: (
            isinstance(parsed, CityReport)
            and parsed.city.lower() == "sydney"
            and parsed.temperature != ""
            and parsed.recommendation != ""
        ),
    },
]


from edge_agent import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    data = {
        "sydney": "25C, clear skies",
        "london": "14C, cloudy with light rain",
        "tokyo": "18C, partly cloudy",
    }
    return data.get(city.lower(), f"No weather data for {city}")


def run_tests() -> None:
    for model_name in MODELS:
        print(f"\n{'═' * 70}")
        print(f"  MODEL: {model_name}")
        print(f"{'═' * 70}")

        provider = OllamaProvider(model=model_name, timeout=300)

        for t in TESTS:
            print(f"\n  Test: {t['label']}")
            print(f"  Q: {t['question']}")

            tools = [get_weather] if t["tools"] else []
            agent = Agent(
                instructions="You are a concise assistant. Use tools when available.",
                provider=provider,
                tools=tools,
            )

            start = time.perf_counter()
            try:
                result = agent.run(
                    t["question"],
                    output_type=t["output_type"],
                    max_turns=10,
                )
                elapsed = time.perf_counter() - start

                print(f"  Raw output: {result.output[:300]}")
                print(f"  Parsed: {result.parsed!r}")

                passed = t["check"](result.parsed) if result.parsed else False
                print(f"  Time: {elapsed:.1f}s  |  {'PASS' if passed else 'FAIL'}")

                if not passed and result.parsed is None:
                    print("  >> parsed is None — model likely returned non-JSON")

            except Exception as exc:
                elapsed = time.perf_counter() - start
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                print(f"  Time: {elapsed:.1f}s  |  FAIL")

    print()


if __name__ == "__main__":
    run_tests()
