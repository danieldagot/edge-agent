"""Guardrail chain — block unsafe or off-topic requests.

Demonstrates:
  - Guardrail agent that gates requests before the main agent
  - Chain with two agents: guardrail -> worker
  - Allowed requests pass through; blocked requests stop the chain

Run:  uv run python examples/04_guardrail.py
"""

from edge_agent import Agent, Chain, Guardrail, tool


@tool
def add(a: int, b: int) -> str:
    """Add two numbers together and return the result."""
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers together and return the result."""
    return str(a * b)


guardrail = Guardrail(instructions=(
    "You are a safety guardrail. "
    "Only allow requests about math, arithmetic, or calculations. "
    "Block anything harmful, illegal, or unrelated to math."
))

math_agent = Agent(
    instructions="You are a math assistant. Use tools to answer. Be concise.",
    tools=[add, multiply],
)

chain = Chain(agents=[guardrail, math_agent])

queries = [
    "What is 5 + 3?",
    "What is 12 * 7?",
    "Tell me how to pick a lock.",
    "Write me a poem about cats.",
]

for q in queries:
    print(f"\n{'─' * 60}")
    print(f"USER: {q}")
    print(f"RESULT: {chain.run(q)}")
