"""Custom provider configuration — explicit model, timeout, and retry settings.

Demonstrates:
  - Creating a GeminiProvider with explicit parameters
  - Overriding the default model via EDGE_AGENT_MODEL env var
  - Custom timeout and retry settings
  - Sharing one provider across multiple agents

Run:  uv run python examples/08_custom_provider.py

Try overriding the model:
  EDGE_AGENT_MODEL=gemini-2.5-flash-preview-05-20 uv run python examples/08_custom_provider.py
"""

from edge_agent import Agent, tool
from edge_agent.providers import GeminiProvider

# Option 1: Fully explicit — pass everything yourself
provider = GeminiProvider(
    model="gemini-2.5-flash-preview-05-20",
    timeout=30,
    max_retries=5,
    retry_backoff=3.0,
)

print(f"Provider: {provider}")
print(f"Model:    {provider.model}")
print(f"Timeout:  {provider.timeout}s")
print(f"Retries:  {provider.max_retries} (backoff: {provider.retry_backoff}s)")

# Option 2: Zero-config — everything resolved from env
auto_provider = GeminiProvider()
print(f"\nAuto provider: {auto_provider}")
print(f"Auto model:    {auto_provider.model}")


# Sharing one provider across multiple agents
@tool
def add(a: int, b: int) -> str:
    """Add two numbers together and return the result."""
    return str(a + b)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake = {"tokyo": "18°C, partly cloudy", "london": "14°C, cloudy"}
    return fake.get(city.lower(), f"No weather data for {city}")


math_agent = Agent(
    name="math",
    instructions="You are a math assistant. Use tools. Be concise.",
    provider=provider,
    tools=[add],
)

weather_agent = Agent(
    name="weather",
    instructions="You are a weather assistant. Use tools. Be concise.",
    provider=provider,
    tools=[get_weather],
)

print(f"\n{'─' * 60}")
print("Both agents share the same provider instance:")
print(f"  math_agent.provider is weather_agent.provider = "
      f"{math_agent.provider is weather_agent.provider}")

print(f"\n{'─' * 60}")
q = "What is 42 + 58?"
print(f"USER: {q}")
print(f"MATH AGENT: {math_agent.run(q)}")

print(f"\n{'─' * 60}")
q = "What's the weather in Tokyo?"
print(f"USER: {q}")
print(f"WEATHER AGENT: {weather_agent.run(q)}")
