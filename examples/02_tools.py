"""Tool usage — define tools with @tool and let the agent call them.

Demonstrates:
  - Different parameter types (int, str, bool)
  - Multi-step tool chaining (the agent calls one tool, then another)
  - Batch queries

Run:  uv run python examples/02_tools.py
"""

from edge_agent import Agent, tool


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
        "sydney": "25°C, clear skies",
    }
    return fake.get(city.lower(), f"No weather data available for {city}")


@tool
def format_greeting(name: str, formal: bool = False) -> str:
    """Greet someone by name, optionally in a formal tone."""
    if formal:
        return f"Good day, {name}. How may I assist you?"
    return f"Hey {name}!"


agent = Agent(
    instructions="You are a helpful assistant. Use the provided tools when needed. Be concise.",
    tools=[add, multiply, get_weather, format_greeting],
)

queries = [
    "What is 7 + 12?",
    "What is 6 * 9?",
    "What's the weather like in London?",
    "Greet Alice formally.",
    "Multiply 3 and 4, then add 10 to the result.",
]

for q in queries:
    print(f"\n{'─' * 60}")
    print(f"USER: {q}")
    print(f"AGENT: {agent.run(q)}")
