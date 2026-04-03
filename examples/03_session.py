"""Interactive REPL — chat with an agent in your terminal.

Demonstrates:
  - Session class wrapping an Agent
  - Conversation memory across turns (ask follow-up questions)
  - Custom prompt and agent label

Run:  uv run python examples/03_session.py

Type 'exit' or 'quit' to end the session.
"""

from edge_agent import Agent, Session, tool


@tool
def add(a: int, b: int) -> str:
    """Add two numbers together and return the result."""
    return str(a + b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers together and return the result."""
    return str(a * b)


agent = Agent(
    name="assistant",
    instructions=(
        "You are a friendly math tutor. Use the provided tools for calculations. "
        "Explain your reasoning step by step. Remember what the user asked earlier."
    ),
    tools=[add, multiply],
)

session = Session(agent, user_label="You", agent_label="Tutor")
session.start()
