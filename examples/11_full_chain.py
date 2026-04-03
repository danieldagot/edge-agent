"""Full chain — all agent types combined in one pipeline.

Demonstrates:
  - Guardrail: blocks unsafe requests at the gate
  - Router: dispatches allowed requests to the right specialist
  - Agent: specialists with scoped tools do the actual work
  - Fallback: specialists that can decline and let the next one try
  - Evaluator: reviews the final output for quality

This is the most complex chain pattern — use it when you need safety,
intelligent routing, specialist tools, graceful degradation, AND
quality review all in one pipeline.

Flow:
  User message → Guardrail → Router → Specialist (Fallback) → Evaluator
                   block?       ↓         fail?                  revise?
                   halt      dispatch    try next              loop back

Run:  uv run python examples/11_full_chain.py
"""

from edge_agent import Agent, Chain, Evaluator, Fallback, Guardrail, Router, tool


# ── tools ────────────────────────────────────────────────────────────────────

@tool
def add(a: int, b: int) -> str:
    """Add two numbers and return the result."""
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers and return the result."""
    return str(a * b)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake = {
        "london": "14°C, cloudy",
        "tokyo": "18°C, partly cloudy",
        "new york": "22°C, sunny",
    }
    return fake.get(city.lower(), f"No data for {city}")


@tool
def search_docs(query: str) -> str:
    """Search the internal documentation."""
    docs = {
        "refund": "Refund policy: full refund within 30 days of purchase.",
        "shipping": "Free shipping on orders over $50. Standard delivery 3-5 days.",
        "warranty": "All products come with a 1-year limited warranty.",
        "returns": "Return items in original packaging within 30 days.",
    }
    for key, value in docs.items():
        if key in query.lower():
            return value
    return "No documentation found for that query."


# ── agents ───────────────────────────────────────────────────────────────────

guardrail = Guardrail(
    name="safety",
    instructions=(
        "You are a safety guardrail for a customer service system.\n"
        "ALLOW requests about: math, weather, product questions, policies.\n"
        "BLOCK requests about: hacking, illegal activity, personal attacks, "
        "or anything harmful."
    ),
)

router = Router(
    name="dispatcher",
    instructions=(
        "Route requests to the right specialist:\n"
        "- math-specialist: calculations, arithmetic\n"
        "- weather-specialist: weather queries\n"
        "- support-specialist: product questions, policies, documentation"
    ),
)

math_specialist = Fallback(
    name="math-specialist",
    instructions=(
        "You are a math specialist. If the request is about math or "
        "arithmetic, solve it using your tools. If it's not about math, "
        "use the fail tool — you cannot help with non-math questions."
    ),
    tools=[add, multiply],
)

weather_specialist = Fallback(
    name="weather-specialist",
    instructions=(
        "You are a weather specialist. If the request is about weather, "
        "answer using your tools. If it's not about weather, use the "
        "fail tool."
    ),
    tools=[get_weather],
)

support_specialist = Agent(
    name="support-specialist",
    instructions=(
        "You are a customer support agent. Search the documentation to "
        "answer policy and product questions. Be helpful and concise."
    ),
    tools=[search_docs],
)

quality_check = Evaluator(
    name="quality-reviewer",
    instructions=(
        "Review the response for quality:\n"
        "- Is it accurate and complete?\n"
        "- Is it concise (not overly verbose)?\n"
        "- Is it helpful and professional in tone?\n\n"
        "Approve if it meets these standards. Otherwise, request a "
        "revision with specific feedback."
    ),
)

# ── chain ────────────────────────────────────────────────────────────────────

# Note: the router will dispatch to one of the specialists.
# If a Fallback specialist can't handle it, the chain moves to the next.
# The evaluator at the end reviews whatever output was produced.

chain = Chain(
    agents=[
        guardrail,
        router,
        math_specialist,
        weather_specialist,
        support_specialist,
        quality_check,
    ],
    max_revisions=1,
)

queries = [
    "What is 25 * 4?",
    "What's the weather in Tokyo?",
    "What is your refund policy?",
    "How do I hack into a database?",
    "How long does shipping take?",
]

for q in queries:
    print(f"\n{'═' * 60}")
    print(f"USER: {q}")
    print(f"RESULT: {chain.run(q)}")
