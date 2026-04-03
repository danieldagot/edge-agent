"""Evaluator chain — review and revise output in a loop.

Demonstrates:
  - Writer agent that produces creative output
  - Evaluator agent that reviews and can request revisions
  - max_revisions controls how many revision rounds are allowed
  - The evaluator uses approve() / revise(feedback) tools injected by the chain

Run:  uv run python examples/06_evaluator.py
"""

from edge_agent import Agent, Chain, Evaluator

writer = Agent(
    name="writer",
    instructions=(
        "You are a creative copywriter. Write a single punchy, memorable "
        "product tagline for the given item. Just output the tagline, nothing else."
    ),
)

evaluator = Evaluator(
    name="editor",
    instructions=(
        "You are a senior copy editor. Review the tagline for:\n"
        "- Clarity: is it immediately understandable?\n"
        "- Impact: does it grab attention?\n"
        "- Brevity: is it concise (under 10 words)?\n\n"
        "If it meets all criteria, approve it. Otherwise, request a revision "
        "with specific feedback on what to improve."
    ),
)

chain = Chain(agents=[writer, evaluator], max_revisions=2)

products = [
    "Wireless noise-cancelling headphones",
    "A reusable water bottle that tracks hydration",
    "An AI-powered code review tool",
]

for product in products:
    print(f"\n{'─' * 60}")
    print(f"PRODUCT: {product}")
    print(f"TAGLINE: {chain.run(product)}")
