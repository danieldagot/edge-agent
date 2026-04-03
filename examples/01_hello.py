"""Minimal hello-world agent — the simplest possible edge-agent script.

Run:  uv run python examples/01_hello.py

Expects GEMINI_API_KEY in a .env file or in the environment.
"""

from edge_agent import Agent

agent = Agent(instructions="You are a helpful assistant. Be concise.")
print(agent.run("What is the capital of Japan?"))
