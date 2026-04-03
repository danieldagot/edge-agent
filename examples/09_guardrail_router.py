"""Guardrail + Router — safety gate before dispatching to specialists.

Demonstrates:
  - Guardrail blocks harmful/off-topic requests before any specialist runs
  - Router dispatches allowed requests to the right specialist
  - Each specialist has its own scoped tools
  - Blocked requests never reach the router or any specialist

Flow:
  User message → Guardrail (block/allow) → Router (route) → Specialist

Run:  uv run python examples/09_guardrail_router.py
"""

from edge_agent import Agent, Chain, Guardrail, Router, tool


# ── tools ────────────────────────────────────────────────────────────────────


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


@tool
def get_forecast(city: str) -> str:
    """Get the weather forecast for the next few days."""
    fake = {
        "london": "Tomorrow: 16°C, sunny. Wednesday: 13°C, showers.",
        "new york": "Tomorrow: 24°C, partly cloudy. Wednesday: 20°C, storms.",
        "tokyo": "Tomorrow: 20°C, clear. Wednesday: 19°C, rain.",
    }
    return fake.get(city.lower(), f"No forecast data for {city}")


# ── agents ───────────────────────────────────────────────────────────────────

guardrail = Guardrail(
    name="safety-gate",
    instructions=(
        "You are a safety guardrail for a math and weather assistant.\n"
        "ALLOW requests about:\n"
        "- Math, arithmetic, calculations\n"
        "- Weather, forecasts, temperature\n\n"
        "BLOCK requests about:\n"
        "- Anything harmful, illegal, or dangerous\n"
        "- Personal opinions, creative writing, or unrelated topics\n\n"
        "When blocking, explain why briefly."
    ),
)

router = Router(
    name="dispatcher",
    instructions=(
        "Route the user's request to the most appropriate specialist.\n"
        "Choose math-specialist for calculations and arithmetic.\n"
        "Choose weather-specialist for weather queries and forecasts."
    ),
)

math_specialist = Agent(
    name="math-specialist",
    instructions="You are a math specialist. Use your tools to compute answers. Be concise.",
    tools=[add, multiply],
)

weather_specialist = Agent(
    name="weather-specialist",
    instructions=(
        "You are a weather specialist. Use your tools to look up weather "
        "conditions and forecasts. Be concise."
    ),
    tools=[get_weather, get_forecast],
)

# ── chain ────────────────────────────────────────────────────────────────────

chain = Chain(agents=[guardrail, router, math_specialist, weather_specialist])

queries = [
    "What is 15 * 8?",                     # allowed → math-specialist
    "What's the weather in London?",        # allowed → weather-specialist
    "Tell me how to hack a computer.",      # blocked by guardrail
    "What is 100 + 200?",                   # allowed → math-specialist
    "Write me a poem about the ocean.",     # blocked by guardrail
    "What's the forecast for Tokyo?",       # allowed → weather-specialist
]

for q in queries:
    print(f"\n{'─' * 60}")
    print(f"USER: {q}")
    print(f"RESULT: {chain.run(q)}")
