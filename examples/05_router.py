"""Router chain — dispatch to the right specialist agent.

Demonstrates:
  - Router agent that picks the best specialist for each query
  - Chain with a router followed by multiple specialist agents
  - Each specialist has its own tools and instructions

Run:  uv run python examples/05_router.py
"""

from edge_agent import Agent, Chain, Router, tool


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
def translate(text: str, target_language: str) -> str:
    """Translate text to the target language (simulated)."""
    translations = {
        "spanish": {"hello": "hola", "goodbye": "adiós", "thank you": "gracias"},
        "french": {"hello": "bonjour", "goodbye": "au revoir", "thank you": "merci"},
        "japanese": {"hello": "こんにちは", "goodbye": "さようなら", "thank you": "ありがとう"},
    }
    lang = translations.get(target_language.lower(), {})
    result = lang.get(text.lower())
    if result:
        return f'"{text}" in {target_language} is "{result}"'
    return f"Sorry, I don't have a translation for \"{text}\" in {target_language}."


router = Router(instructions=(
    "Route the user's request to the most appropriate specialist. "
    "Choose math-specialist for calculations, weather-specialist for "
    "weather queries, or translator for translation requests."
))

math_specialist = Agent(
    name="math-specialist",
    instructions="You are a math specialist. Use tools to compute answers. Be concise.",
    tools=[add, multiply],
)

weather_specialist = Agent(
    name="weather-specialist",
    instructions="You are a weather specialist. Use tools to look up weather. Be concise.",
    tools=[get_weather],
)

translator_agent = Agent(
    name="translator",
    instructions="You are a translator. Use tools to translate text. Be concise.",
    tools=[translate],
)

chain = Chain(agents=[router, math_specialist, weather_specialist, translator_agent])

queries = [
    "What is 12 + 7?",
    "What's the weather in Tokyo?",
    "How do you say 'hello' in French?",
    "What is 8 * 6?",
]

for q in queries:
    print(f"\n{'─' * 60}")
    print(f"USER: {q}")
    print(f"RESULT: {chain.run(q)}")
