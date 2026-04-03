"""Content pipeline — guardrail + writer + evaluator.

Demonstrates:
  - Guardrail filters unsafe prompts before content generation begins
  - Writer agent produces content (no tools needed — pure text generation)
  - Evaluator reviews and can request revisions with specific feedback
  - pass_original=False so the evaluator sees the writer's output, not the raw prompt

Flow:
  User message → Guardrail (block/allow) → Writer (generate) → Evaluator (approve/revise)
                                              ↑_______revise loop_________|

Run:  uv run python examples/10_pipeline.py
"""

from edge_agent import Agent, Chain, Evaluator, Guardrail


# ── agents ───────────────────────────────────────────────────────────────────

guardrail = Guardrail(
    name="content-filter",
    instructions=(
        "You are a content safety filter for a professional copywriting service.\n"
        "ALLOW requests to write:\n"
        "- Product descriptions, taglines, marketing copy\n"
        "- Professional emails, announcements\n"
        "- Technical summaries, documentation\n\n"
        "BLOCK requests to write:\n"
        "- Anything deceptive, misleading, or fraudulent\n"
        "- Hate speech, harassment, or harmful content\n"
        "- Content that impersonates real people\n\n"
        "When blocking, explain the policy violation briefly."
    ),
)

writer = Agent(
    name="copywriter",
    instructions=(
        "You are a senior copywriter. Write exactly what is requested.\n"
        "- Keep it punchy and professional\n"
        "- Match the tone to the product/context\n"
        "- Output only the copy, no commentary or meta-text"
    ),
)

evaluator = Evaluator(
    name="editor",
    instructions=(
        "You are a senior copy editor reviewing marketing content.\n"
        "Evaluate the copy on these criteria:\n"
        "1. Clarity — is the message immediately understandable?\n"
        "2. Impact — does it grab attention and stick in memory?\n"
        "3. Brevity — is it concise without being vague?\n"
        "4. Tone — does it feel professional and on-brand?\n\n"
        "If ALL criteria are met, approve the copy.\n"
        "Otherwise, request a revision with specific, actionable feedback "
        "on exactly what to improve."
    ),
)

# ── chain ────────────────────────────────────────────────────────────────────

chain = Chain(
    agents=[guardrail, writer, evaluator],
    max_revisions=2,
)

prompts = [
    "Write a tagline for a smart water bottle that tracks hydration",
    "Write a product description for wireless earbuds with 40-hour battery life",
    "Write a fake testimonial pretending to be Elon Musk endorsing my product",
    "Write a one-line email subject for a flash sale on running shoes",
]

for prompt in prompts:
    print(f"\n{'─' * 60}")
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {chain.run(prompt)}")
