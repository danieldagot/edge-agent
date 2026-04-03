"""Fallback chain — try specialists in order, fall back to a generalist.

Demonstrates:
  - Fallback agents that signal failure on requests they can't handle
  - The chain automatically tries the next agent when one fails
  - The last agent (a generalist) catches everything

Run:  uv run python examples/07_fallback.py
"""

from edge_agent import Agent, Chain, Fallback, tool


@tool
def add(a: int, b: int) -> str:
    """Add two numbers together and return the result."""
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers together and return the result."""
    return str(a * b)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake = {
        "london": "14°C, cloudy with a chance of rain",
        "new york": "22°C, sunny",
        "tokyo": "18°C, partly cloudy",
    }
    return fake.get(city.lower(), f"No weather data available for {city}")


math_only = Fallback(
    name="math-only",
    instructions=(
        "You are a math-only agent. If the request is about math or "
        "arithmetic, answer it using the tools. If it's about anything "
        "else, you cannot handle it — use the fail tool."
    ),
    tools=[add, multiply],
)

weather_only = Fallback(
    name="weather-only",
    instructions=(
        "You are a weather-only agent. If the request is about weather, "
        "answer it using the tools. If it's about anything else, you "
        "cannot handle it — use the fail tool."
    ),
    tools=[get_weather],
)

generalist = Agent(
    name="generalist",
    instructions=(
        "You are a helpful general-purpose assistant. "
        "Answer any question concisely."
    ),
)

chain = Chain(agents=[math_only, weather_only, generalist])

queries = [
    "What is 9 + 4?",
    "What's the weather in London?",
    "What is the capital of France?",
    "Who wrote Hamlet?",
]

for q in queries:
    print(f"\n{'─' * 60}")
    print(f"USER: {q}")
    print(f"RESULT: {chain.run(q)}")
