"""Tests for Chain with all agent types and control flow."""

from __future__ import annotations

import pytest

from edge_agent import Agent, Chain, Evaluator, Fallback, Guardrail, Router
from edge_agent.types import Message, ToolCall

from tests.conftest import MockProvider


def _text(content: str) -> Message:
    return Message(role="assistant", content=content)


def _tool_call(name: str, args: dict | None = None) -> Message:
    return Message(
        role="assistant",
        tool_calls=[ToolCall(name=name, arguments=args or {}, id="c1")],
    )


# ── basic chain ─────────────────────────────────────────────────────────────


class TestChainBasic:
    def test_empty_chain_raises(self):
        with pytest.raises(ValueError, match="at least one agent"):
            Chain(agents=[])

    def test_single_agent(self):
        provider = MockProvider([_text("hello")])
        agent = Agent(name="a", instructions="x", provider=provider)
        chain = Chain(agents=[agent])

        result = chain.run("hi")
        assert result.output == "hello"

    def test_two_agents_sequential(self):
        p1 = MockProvider([_text("step-1")])
        p2 = MockProvider([_text("step-2")])
        a1 = Agent(name="a1", instructions="x", provider=p1)
        a2 = Agent(name="a2", instructions="x", provider=p2)
        chain = Chain(agents=[a1, a2])

        result = chain.run("go")
        assert result.output == "step-2"

    def test_pass_original_true_sends_original_message(self):
        p1 = MockProvider([_text("step-1")])
        p2 = MockProvider([_text("step-2")])
        a1 = Agent(name="a1", instructions="x", provider=p1)
        a2 = Agent(name="a2", instructions="x", provider=p2)
        chain = Chain(agents=[a1, a2], pass_original=True)

        chain.run("original question")
        user_msg = p2.call_log[0][0][1]
        assert user_msg.content == "original question"

    def test_pass_original_false_forwards_previous_output(self):
        p1 = MockProvider([_text("transformed")])
        p2 = MockProvider([_text("final")])
        a1 = Agent(name="a1", instructions="x", provider=p1)
        a2 = Agent(name="a2", instructions="x", provider=p2)
        chain = Chain(agents=[a1, a2], pass_original=False)

        chain.run("start")
        user_msg = p2.call_log[0][0][1]
        assert user_msg.content == "transformed"


# ── guardrail ───────────────────────────────────────────────────────────────


class TestChainGuardrail:
    def test_block_stops_chain(self):
        guard_provider = MockProvider([
            _tool_call("block", {"reason": "not allowed"}),
            _text("Blocked: not allowed"),
        ])
        main_provider = MockProvider([_text("should not reach")])

        guard = Guardrail(name="guard", instructions="x", provider=guard_provider)
        main = Agent(name="main", instructions="x", provider=main_provider)
        chain = Chain(agents=[guard, main])

        result = chain.run("bad request")
        assert "Blocked" in result.output or "not allowed" in result.output
        assert len(main_provider.call_log) == 0

    def test_allow_continues_chain(self):
        guard_provider = MockProvider([
            _tool_call("allow"),
            _text("Allowed."),
        ])
        main_provider = MockProvider([_text("main result")])

        guard = Guardrail(name="guard", instructions="x", provider=guard_provider)
        main = Agent(name="main", instructions="x", provider=main_provider)
        chain = Chain(agents=[guard, main])

        result = chain.run("good request")
        assert result.output == "main result"
        assert len(main_provider.call_log) == 1


# ── router ──────────────────────────────────────────────────────────────────


class TestChainRouter:
    def test_routes_to_correct_agent(self):
        router_provider = MockProvider([
            _tool_call("route", {"agent_name": "specialist-b", "reason": "matches"}),
            _text("Routing to specialist-b"),
        ])
        pa = MockProvider([_text("A result")])
        pb = MockProvider([_text("B result")])

        router = Router(name="router", instructions="x", provider=router_provider)
        agent_a = Agent(name="specialist-a", instructions="x", provider=pa)
        agent_b = Agent(name="specialist-b", instructions="x", provider=pb)
        chain = Chain(agents=[router, agent_a, agent_b])

        result = chain.run("go to B")
        assert result.output == "B result"
        assert len(pa.call_log) == 0
        assert len(pb.call_log) == 1

    def test_unknown_route_continues_normally(self):
        router_provider = MockProvider([
            _tool_call("route", {"agent_name": "nonexistent", "reason": "guess"}),
            _text("Routing attempt"),
        ])
        pa = MockProvider([_text("A result")])

        router = Router(name="router", instructions="x", provider=router_provider)
        agent_a = Agent(name="a", instructions="x", provider=pa)
        chain = Chain(agents=[router, agent_a])

        result = chain.run("go somewhere")
        assert result.output == "A result"


# ── evaluator ───────────────────────────────────────────────────────────────


class TestChainEvaluator:
    def test_approve_keeps_output(self):
        writer_provider = MockProvider([_text("great tagline")])
        eval_provider = MockProvider([
            _tool_call("approve"),
            _text("Approved."),
        ])

        writer = Agent(name="writer", instructions="x", provider=writer_provider)
        evl = Evaluator(name="eval", instructions="x", provider=eval_provider)
        chain = Chain(agents=[writer, evl])

        result = chain.run("write something")
        assert result.output == "great tagline"

    def test_revise_loops_back(self):
        writer_provider = MockProvider([
            _text("draft 1"),
            _text("draft 2"),
        ])
        eval_provider = MockProvider([
            _tool_call("revise", {"feedback": "too bland"}),
            _text("Revision requested"),
            _tool_call("approve"),
            _text("Approved."),
        ])

        writer = Agent(name="writer", instructions="x", provider=writer_provider)
        evl = Evaluator(name="eval", instructions="x", provider=eval_provider)
        chain = Chain(agents=[writer, evl], max_revisions=3)

        result = chain.run("write something")
        assert result.output == "draft 2"
        assert len(writer_provider.call_log) == 2

    def test_max_revisions_stops_loop(self):
        writer_provider = MockProvider([
            _text("draft 1"),
            _text("draft 2"),
            _text("draft 3"),
        ])
        eval_provider = MockProvider([
            _tool_call("revise", {"feedback": "no"}),
            _text("Nope"),
            _tool_call("revise", {"feedback": "still no"}),
            _text("Nope"),
            _tool_call("revise", {"feedback": "still still no"}),
            _text("Nope"),
        ])

        writer = Agent(name="writer", instructions="x", provider=writer_provider)
        evl = Evaluator(name="eval", instructions="x", provider=eval_provider)
        chain = Chain(agents=[writer, evl], max_revisions=2)

        result = chain.run("write something")
        assert result.output == "draft 3"


# ── fallback ────────────────────────────────────────────────────────────────


class TestChainFallback:
    def test_success_stops_chain(self):
        p1 = MockProvider([_text("handled it")])
        p2 = MockProvider([_text("backup")])

        fb1 = Fallback(name="first", instructions="x", provider=p1)
        fb2 = Agent(name="second", instructions="x", provider=p2)
        chain = Chain(agents=[fb1, fb2])

        result = chain.run("something")
        assert result.output == "handled it"
        assert len(p2.call_log) == 0

    def test_fail_moves_to_next(self):
        p1 = MockProvider([
            _tool_call("fail", {"reason": "cannot do this"}),
            _text("Failed"),
        ])
        p2 = MockProvider([_text("backup handled it")])

        fb1 = Fallback(name="first", instructions="x", provider=p1)
        fb2 = Agent(name="second", instructions="x", provider=p2)
        chain = Chain(agents=[fb1, fb2])

        result = chain.run("something")
        assert result.output == "backup handled it"
        assert len(p2.call_log) == 1

    def test_multiple_fallbacks_cascade(self):
        """Three fallbacks in a row — first two fail, third succeeds."""
        p1 = MockProvider([
            _tool_call("fail", {"reason": "not my domain"}),
            _text("Failed"),
        ])
        p2 = MockProvider([
            _tool_call("fail", {"reason": "not mine either"}),
            _text("Failed"),
        ])
        p3 = MockProvider([_text("third agent handled it")])

        fb1 = Fallback(name="fb1", instructions="x", provider=p1)
        fb2 = Fallback(name="fb2", instructions="x", provider=p2)
        fb3 = Fallback(name="fb3", instructions="x", provider=p3)
        chain = Chain(agents=[fb1, fb2, fb3])

        result = chain.run("something")
        assert result.output == "third agent handled it"
        assert len(p1.call_log) == 2
        assert len(p2.call_log) == 2
        assert len(p3.call_log) == 1


# ── combined chains ─────────────────────────────────────────────────────────


class TestChainGuardrailRouter:
    """Guardrail + Router: safety gate before dispatching."""

    def test_block_prevents_routing(self):
        guard_p = MockProvider([
            _tool_call("block", {"reason": "unsafe"}),
            _text("Blocked: unsafe"),
        ])
        router_p = MockProvider([_text("should not reach")])
        specialist_p = MockProvider([_text("should not reach")])

        guard = Guardrail(name="guard", instructions="x", provider=guard_p)
        router = Router(name="router", instructions="x", provider=router_p)
        spec = Agent(name="spec", instructions="x", provider=specialist_p)
        chain = Chain(agents=[guard, router, spec])

        result = chain.run("bad request")
        assert "Blocked" in result.output or "unsafe" in result.output
        assert len(router_p.call_log) == 0
        assert len(specialist_p.call_log) == 0

    def test_allow_then_route_to_specialist(self):
        guard_p = MockProvider([
            _tool_call("allow"),
            _text("Allowed."),
        ])
        router_p = MockProvider([
            _tool_call("route", {"agent_name": "math", "reason": "math query"}),
            _text("Routing to math"),
        ])
        math_p = MockProvider([_text("42")])
        weather_p = MockProvider([_text("should not reach")])

        guard = Guardrail(name="guard", instructions="x", provider=guard_p)
        router = Router(name="router", instructions="x", provider=router_p)
        math_agent = Agent(name="math", instructions="x", provider=math_p)
        weather_agent = Agent(name="weather", instructions="x", provider=weather_p)
        chain = Chain(agents=[guard, router, math_agent, weather_agent])

        result = chain.run("what is 6*7")
        assert result.output == "42"
        assert len(weather_p.call_log) == 0

    def test_allow_then_route_skips_other_specialists(self):
        guard_p = MockProvider([
            _tool_call("allow"),
            _text("Allowed."),
        ])
        router_p = MockProvider([
            _tool_call("route", {"agent_name": "weather", "reason": "weather query"}),
            _text("Routing to weather"),
        ])
        math_p = MockProvider([_text("should not reach")])
        weather_p = MockProvider([_text("sunny 25C")])

        guard = Guardrail(name="guard", instructions="x", provider=guard_p)
        router = Router(name="router", instructions="x", provider=router_p)
        math_agent = Agent(name="math", instructions="x", provider=math_p)
        weather_agent = Agent(name="weather", instructions="x", provider=weather_p)
        chain = Chain(agents=[guard, router, math_agent, weather_agent])

        result = chain.run("weather in London")
        assert result.output == "sunny 25C"
        assert len(math_p.call_log) == 0


class TestChainGuardrailEvaluator:
    """Guardrail + Writer + Evaluator: safety-gated content pipeline."""

    def test_block_prevents_writing(self):
        guard_p = MockProvider([
            _tool_call("block", {"reason": "inappropriate"}),
            _text("Blocked"),
        ])
        writer_p = MockProvider([_text("should not reach")])
        eval_p = MockProvider([_text("should not reach")])

        guard = Guardrail(name="guard", instructions="x", provider=guard_p)
        writer = Agent(name="writer", instructions="x", provider=writer_p)
        evl = Evaluator(name="eval", instructions="x", provider=eval_p)
        chain = Chain(agents=[guard, writer, evl])

        result = chain.run("write something bad")
        assert "Blocked" in result.output or "inappropriate" in result.output
        assert len(writer_p.call_log) == 0
        assert len(eval_p.call_log) == 0

    def test_allow_then_write_then_approve(self):
        guard_p = MockProvider([
            _tool_call("allow"),
            _text("Allowed."),
        ])
        writer_p = MockProvider([_text("great copy")])
        eval_p = MockProvider([
            _tool_call("approve"),
            _text("Approved."),
        ])

        guard = Guardrail(name="guard", instructions="x", provider=guard_p)
        writer = Agent(name="writer", instructions="x", provider=writer_p)
        evl = Evaluator(name="eval", instructions="x", provider=eval_p)
        chain = Chain(agents=[guard, writer, evl])

        result = chain.run("write a tagline")
        assert result.output == "great copy"

    def test_allow_then_write_then_revise_then_approve(self):
        guard_p = MockProvider([
            _tool_call("allow"),
            _text("Allowed."),
        ])
        writer_p = MockProvider([
            _text("draft 1"),
            _text("draft 2"),
        ])
        eval_p = MockProvider([
            _tool_call("revise", {"feedback": "too generic"}),
            _text("Revision requested"),
            _tool_call("approve"),
            _text("Approved."),
        ])

        guard = Guardrail(name="guard", instructions="x", provider=guard_p)
        writer = Agent(name="writer", instructions="x", provider=writer_p)
        evl = Evaluator(name="eval", instructions="x", provider=eval_p)
        chain = Chain(agents=[guard, writer, evl], max_revisions=2)

        result = chain.run("write a tagline")
        assert result.output == "draft 2"
        assert len(writer_p.call_log) == 2


class TestChainFallbackEvaluator:
    """Fallback cascade + Evaluator: try specialists, then review the result."""

    def test_first_fallback_succeeds_then_evaluator_approves(self):
        fb_p = MockProvider([_text("specialist result")])
        eval_p = MockProvider([
            _tool_call("approve"),
            _text("Approved."),
        ])

        fb = Fallback(name="specialist", instructions="x", provider=fb_p)
        evl = Evaluator(name="eval", instructions="x", provider=eval_p)
        chain = Chain(agents=[fb, evl])

        result = chain.run("question")
        assert result.output == "specialist result"

    def test_fallback_fails_then_agent_answers_then_evaluator_approves(self):
        fb_p = MockProvider([
            _tool_call("fail", {"reason": "not my area"}),
            _text("Failed"),
        ])
        agent_p = MockProvider([_text("generalist result")])
        eval_p = MockProvider([
            _tool_call("approve"),
            _text("Approved."),
        ])

        fb = Fallback(name="specialist", instructions="x", provider=fb_p)
        agent = Agent(name="generalist", instructions="x", provider=agent_p)
        evl = Evaluator(name="eval", instructions="x", provider=eval_p)
        chain = Chain(agents=[fb, agent, evl])

        result = chain.run("general question")
        assert result.output == "generalist result"
        assert len(agent_p.call_log) == 1


class TestChainToolScoping:
    """Verify that each agent only sees its own tools in API calls."""

    def test_router_only_sees_route_tool(self):
        from edge_agent import tool

        @tool
        def add(a: int, b: int) -> str:
            """Add numbers."""
            return str(a + b)

        router_p = MockProvider([
            _tool_call("route", {"agent_name": "math", "reason": "math"}),
            _text("Routing"),
        ])
        math_p = MockProvider([_text("42")])

        router = Router(name="router", instructions="x", provider=router_p)
        math_agent = Agent(name="math", instructions="x", provider=math_p, tools=[add])
        chain = Chain(agents=[router, math_agent])
        chain.run("what is 6*7")

        router_tools = router_p.call_log[0][1]
        assert router_tools is not None
        router_tool_names = [t.name for t in router_tools]
        assert "route" in router_tool_names
        assert "add" not in router_tool_names

    def test_specialist_only_sees_own_tools(self):
        from edge_agent import tool

        @tool
        def calc(x: int) -> str:
            """Calculate."""
            return str(x)

        @tool
        def lookup(q: str) -> str:
            """Look up."""
            return q

        router_p = MockProvider([
            _tool_call("route", {"agent_name": "calc-agent", "reason": "calc"}),
            _text("Routing"),
        ])
        calc_p = MockProvider([_text("result")])
        lookup_p = MockProvider([_text("should not run")])

        router = Router(name="router", instructions="x", provider=router_p)
        calc_agent = Agent(name="calc-agent", instructions="x", provider=calc_p, tools=[calc])
        lookup_agent = Agent(name="lookup-agent", instructions="x", provider=lookup_p, tools=[lookup])
        chain = Chain(agents=[router, calc_agent, lookup_agent])
        chain.run("calculate something")

        calc_tools = calc_p.call_log[0][1]
        assert calc_tools is not None
        calc_tool_names = [t.name for t in calc_tools]
        assert "calc" in calc_tool_names
        assert "lookup" not in calc_tool_names
        assert "route" not in calc_tool_names

    def test_guardrail_only_sees_guard_tools(self):
        guard_p = MockProvider([
            _tool_call("allow"),
            _text("Allowed"),
        ])
        agent_p = MockProvider([_text("result")])

        from edge_agent import tool

        @tool
        def my_tool(x: str) -> str:
            """A tool."""
            return x

        guard = Guardrail(name="guard", instructions="x", provider=guard_p)
        agent = Agent(name="worker", instructions="x", provider=agent_p, tools=[my_tool])
        chain = Chain(agents=[guard, agent])
        chain.run("test")

        guard_tools = guard_p.call_log[0][1]
        assert guard_tools is not None
        guard_tool_names = [t.name for t in guard_tools]
        assert "block" in guard_tool_names
        assert "allow" in guard_tool_names
        assert "my_tool" not in guard_tool_names


# ── RunResult tracing ───────────────────────────────────────────────────────


class TestChainRunResultTracing:
    def test_single_agent_chain_has_one_step(self):
        provider = MockProvider([_text("hello")])
        agent = Agent(name="solo", instructions="x", provider=provider)
        chain = Chain(agents=[agent])

        result = chain.run("hi")
        assert len(result.steps) == 1
        assert result.steps[0].agent_name == "solo"
        assert result.steps[0].agent_type == "agent"
        assert result.steps[0].output == "hello"

    def test_two_agents_have_two_steps(self):
        p1 = MockProvider([_text("step-1")])
        p2 = MockProvider([_text("step-2")])
        a1 = Agent(name="first", instructions="x", provider=p1)
        a2 = Agent(name="second", instructions="x", provider=p2)
        chain = Chain(agents=[a1, a2])

        result = chain.run("go")
        assert len(result.steps) == 2
        assert result.steps[0].agent_name == "first"
        assert result.steps[1].agent_name == "second"

    def test_guardrail_block_records_step(self):
        guard_p = MockProvider([
            _tool_call("block", {"reason": "unsafe"}),
            _text("Blocked"),
        ])
        main_p = MockProvider([_text("should not reach")])

        guard = Guardrail(name="guard", instructions="x", provider=guard_p)
        main = Agent(name="main", instructions="x", provider=main_p)
        chain = Chain(agents=[guard, main])

        result = chain.run("bad")
        assert len(result.steps) == 1
        assert result.steps[0].agent_name == "guard"
        assert result.steps[0].agent_type == "guardrail"
        assert len(result.steps[0].tools_used) == 1
        assert result.steps[0].tools_used[0].name == "block"

    def test_router_records_both_steps(self):
        router_p = MockProvider([
            _tool_call("route", {"agent_name": "worker", "reason": "match"}),
            _text("Routing"),
        ])
        worker_p = MockProvider([_text("done")])

        router = Router(name="router", instructions="x", provider=router_p)
        worker = Agent(name="worker", instructions="x", provider=worker_p)
        chain = Chain(agents=[router, worker])

        result = chain.run("go")
        assert len(result.steps) == 2
        assert result.steps[0].agent_name == "router"
        assert result.steps[0].tools_used[0].name == "route"
        assert result.steps[1].agent_name == "worker"

    def test_evaluator_records_revision_steps(self):
        writer_p = MockProvider([_text("draft 1"), _text("draft 2")])
        eval_p = MockProvider([
            _tool_call("revise", {"feedback": "improve"}),
            _text("Revise"),
            _tool_call("approve"),
            _text("Approved"),
        ])

        writer = Agent(name="writer", instructions="x", provider=writer_p)
        evl = Evaluator(name="eval", instructions="x", provider=eval_p)
        chain = Chain(agents=[writer, evl], max_revisions=3)

        result = chain.run("write")
        writer_steps = [s for s in result.steps if s.agent_name == "writer"]
        eval_steps = [s for s in result.steps if s.agent_name == "eval"]
        assert len(writer_steps) == 2
        assert len(eval_steps) == 2

    def test_str_on_chain_result(self):
        provider = MockProvider([_text("chain output")])
        agent = Agent(name="a", instructions="x", provider=provider)
        chain = Chain(agents=[agent])

        result = chain.run("go")
        assert str(result) == "chain output"
