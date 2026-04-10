"""Ollama model benchmark — compare tool-calling quality and speed.

Tests each model on 9 queries of increasing complexity, measuring:
  - Response time per query
  - Correctness score (automated checks per question)
  - Structured output (JSON dataclass parsing)

Run:
  uv run python examples/benchmark_ollama.py
"""

import time
from dataclasses import dataclass
from typing import Any

from edge_agent import Agent, tool
from edge_agent.providers import OllamaProvider

MODELS = [
    "gemma4:e4b",
    "qwen3.5:9b",
    "qwen3:4b-instruct",
    "llama3.1:8b",
]


# ── tools ───────────────────────────────────────────────────────────────────


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
    data = {
        "london": "14C, cloudy with light rain",
        "new york": "22C, sunny",
        "tokyo": "18C, partly cloudy",
        "sydney": "25C, clear skies",
        "paris": "16C, overcast",
        "berlin": "11C, windy and cold",
    }
    return data.get(city.lower(), f"No weather data available for {city}")


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another."""
    rates = {
        ("usd", "eur"): 0.92, ("eur", "usd"): 1.09,
        ("usd", "gbp"): 0.79, ("gbp", "usd"): 1.27,
        ("usd", "jpy"): 149.50, ("jpy", "usd"): 0.0067,
        ("eur", "gbp"): 0.86, ("gbp", "eur"): 1.16,
    }
    key = (from_currency.lower(), to_currency.lower())
    rate = rates.get(key)
    if rate is None:
        return f"No conversion rate available for {from_currency} to {to_currency}"
    converted = round(amount * rate, 2)
    return f"{amount} {from_currency.upper()} = {converted} {to_currency.upper()}"


@tool
def search_contacts(name: str) -> str:
    """Search for a contact by name and return their info."""
    contacts = {
        "alice": "Alice Johnson — alice@example.com — +1-555-0101 — New York",
        "bob": "Bob Smith — bob@example.com — +1-555-0102 — London",
        "charlie": "Charlie Brown — charlie@example.com — +81-90-1234-5678 — Tokyo",
        "diana": "Diana Prince — diana@example.com — +33-6-1234-5678 — Paris",
    }
    return contacts.get(name.lower(), f"No contact found for '{name}'")


@tool
def calculate_tip(bill_amount: float, tip_percent: float) -> str:
    """Calculate the tip and total for a restaurant bill."""
    tip = round(bill_amount * tip_percent / 100, 2)
    total = round(bill_amount + tip, 2)
    return f"Bill: ${bill_amount}, Tip ({tip_percent}%): ${tip}, Total: ${total}"


ALL_TOOLS = [add, multiply, get_weather, convert_currency, search_contacts, calculate_tip]


# ── structured output type ─────────────────────────────────────────────────


@dataclass
class CityReport:
    city: str
    temperature: str
    conditions: str
    recommendation: str


# ── queries and correctness checks ─────────────────────────────────────────

Query = dict[str, Any]

QUERIES: list[Query] = [
    {
        "question": "What is 7 + 12?",
        "check": lambda out: "19" in out,
        "label": "simple add",
    },
    {
        "question": "What's the weather like in Tokyo?",
        "check": lambda out: "18" in out.lower() or "partly cloudy" in out.lower(),
        "label": "single tool",
    },
    {
        "question": "What is 6 times 9?",
        "check": lambda out: "54" in out,
        "label": "multiply",
    },
    {
        "question": "Convert 100 USD to EUR.",
        "check": lambda out: "92" in out,
        "label": "currency",
    },
    {
        "question": "Look up Charlie's contact info and tell me what city he lives in.",
        "check": lambda out: "tokyo" in out.lower(),
        "label": "contact lookup",
    },
    {
        "question": "Add 5 and 6, then tell me the weather in London.",
        "check": lambda out: "11" in out and ("14" in out or "cloudy" in out.lower() or "rain" in out.lower()),
        "label": "2-tool chain",
    },
    {
        "question": (
            "I have a $85 dinner bill. Calculate a 20% tip, "
            "then convert the total from USD to GBP."
        ),
        "check": lambda out: "102" in out and ("80" in out or "gbp" in out.lower()),
        "label": "3-step chain",
    },
    {
        "question": (
            "Find Diana's contact, check the weather in her city, "
            "then convert 250 EUR to USD and tell me all three results."
        ),
        "check": lambda out: (
            "diana" in out.lower()
            and ("paris" in out.lower() or "16" in out or "overcast" in out.lower())
            and "272" in out
        ),
        "label": "3-tool fan-out",
    },
    {
        "question": (
            "Get the weather in Sydney and return a structured city report "
            "with a recommendation on what to wear."
        ),
        "check": lambda out, parsed=None: (
            parsed is not None
            and isinstance(parsed, CityReport)
            and parsed.city.lower() == "sydney"
            and parsed.temperature != ""
            and parsed.conditions != ""
            and parsed.recommendation != ""
        ),
        "label": "structured",
        "output_type": CityReport,
    },
]


def run_benchmark() -> None:
    results: list[dict[str, Any]] = []

    for model_name in MODELS:
        print(f"\n{'═' * 74}")
        print(f"  MODEL: {model_name}")
        print(f"{'═' * 74}")

        try:
            provider = OllamaProvider(model=model_name, timeout=300)
        except Exception as exc:
            print(f"  SKIP — failed to create provider: {exc}")
            results.append({"model": model_name, "error": str(exc)})
            continue

        agent = Agent(
            instructions=(
                "You are a concise assistant. "
                "ALWAYS use the provided tools to answer questions — "
                "never guess or calculate manually. "
                "When a question requires multiple steps, call each tool "
                "separately and combine the results."
            ),
            provider=provider,
            tools=ALL_TOOLS,
        )

        timings: list[float] = []
        checks: list[bool] = []

        for q in QUERIES:
            print(f"\n  Q [{q['label']}]: {q['question']}")
            output_type = q.get("output_type")
            start = time.perf_counter()
            try:
                result = agent.run(q["question"], max_turns=15, output_type=output_type)
                elapsed = time.perf_counter() - start
                output = str(result)
                tools_used = [t.name for t in result.steps[0].tools_used] if result.steps else []

                if output_type is not None:
                    passed = q["check"](output, parsed=result.parsed)
                    parsed_repr = repr(result.parsed) if result.parsed else "None"
                    print(f"  Parsed: {parsed_repr[:200]}")
                else:
                    passed = q["check"](output)

                print(f"  A: {output[:200]}{'…' if len(output) > 200 else ''}")
                print(f"  Tools: {tools_used or 'none'}")
                print(f"  Time: {elapsed:.1f}s  |  {'PASS' if passed else 'FAIL'}")
                timings.append(elapsed)
                checks.append(passed)
            except Exception as exc:
                elapsed = time.perf_counter() - start
                print(f"  ERROR: {exc} ({elapsed:.1f}s)")
                timings.append(elapsed)
                checks.append(False)

        score = sum(checks)
        results.append({
            "model": model_name,
            "timings": timings,
            "total": sum(timings),
            "checks": checks,
            "score": score,
        })

    # ── summary table ───────────────────────────────────────────────────
    labels = [q["label"] for q in QUERIES]
    col_w = 8

    print(f"\n\n{'═' * 74}")
    print("  SUMMARY")
    print(f"{'═' * 74}\n")

    header = f"  {'Model':<22}"
    for lb in labels:
        header += f" {lb[:col_w]:>{col_w}}"
    header += f" {'Total':>7} {'Score':>6}"
    print(header)
    print(f"  {'─' * 22}" + f" {'─' * col_w}" * len(labels) + f" {'─' * 7} {'─' * 6}")

    for r in results:
        if "error" in r:
            print(f"  {r['model']:<22} SKIP")
            continue
        row = f"  {r['model']:<22}"
        for i, t in enumerate(r["timings"]):
            mark = "+" if r["checks"][i] else "X"
            row += f" {mark}{t:.0f}s".rjust(col_w + 1)
            if len(row.split('\n')[-1]) > 200:
                break
        row += f" {r['total']:>6.0f}s {r['score']}/{len(QUERIES):>4}"
        print(row)

    print()
    best = max((r for r in results if "error" not in r), key=lambda r: (r["score"], -r["total"]), default=None)
    if best:
        print(f"  Winner: {best['model']} — {best['score']}/{len(QUERIES)} correct, {best['total']:.0f}s total")
    print()


if __name__ == "__main__":
    run_benchmark()
