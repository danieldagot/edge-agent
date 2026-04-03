"""Tests for agent type validation and typed subclasses."""

from __future__ import annotations

import pytest

from edge_agent import Agent, Evaluator, Fallback, Guardrail, Router
from edge_agent.agent import AgentType, _VALID_AGENT_TYPES
from edge_agent.types import Message

from tests.conftest import MockProvider


# ── AgentType validation ────────────────────────────────────────────────────


class TestAgentTypeValidation:
    def _provider(self) -> MockProvider:
        return MockProvider([Message(role="assistant", content="ok")])

    def test_default_type_is_agent(self):
        agent = Agent(
            name="a", instructions="x", provider=self._provider(),
        )
        assert agent.agent_type == "agent"

    @pytest.mark.parametrize("agent_type", sorted(_VALID_AGENT_TYPES))
    def test_valid_types_accepted(self, agent_type: str):
        agent = Agent(
            name="a",
            instructions="x",
            provider=self._provider(),
            agent_type=agent_type,  # type: ignore[arg-type]
        )
        assert agent.agent_type == agent_type

    @pytest.mark.parametrize("bad_type", ["guardian", "route", "eval", "fb", "", "GUARDRAIL"])
    def test_invalid_type_raises(self, bad_type: str):
        with pytest.raises(ValueError, match="Invalid agent_type"):
            Agent(
                name="a",
                instructions="x",
                provider=self._provider(),
                agent_type=bad_type,  # type: ignore[arg-type]
            )


# ── typed subclasses ────────────────────────────────────────────────────────


class TestGuardrail:
    def test_type_is_guardrail(self):
        g = Guardrail(
            name="g",
            instructions="block bad stuff",
            provider=MockProvider([Message(role="assistant", content="ok")]),
        )
        assert g.agent_type == "guardrail"
        assert isinstance(g, Agent)

    def test_has_no_agent_type_kwarg(self):
        """Subclasses should not accept agent_type — it's set automatically."""
        with pytest.raises(TypeError):
            Guardrail(
                name="g",
                instructions="x",
                provider=MockProvider([Message(role="assistant", content="ok")]),
                agent_type="agent",  # type: ignore[call-arg]
            )


class TestRouter:
    def test_type_is_router(self):
        r = Router(
            name="r",
            instructions="route stuff",
            provider=MockProvider([Message(role="assistant", content="ok")]),
        )
        assert r.agent_type == "router"
        assert isinstance(r, Agent)

    def test_rejects_agent_type_kwarg(self):
        with pytest.raises(TypeError):
            Router(
                name="r",
                instructions="x",
                provider=MockProvider([Message(role="assistant", content="ok")]),
                agent_type="agent",  # type: ignore[call-arg]
            )


class TestEvaluator:
    def test_type_is_evaluator(self):
        e = Evaluator(
            name="e",
            instructions="evaluate stuff",
            provider=MockProvider([Message(role="assistant", content="ok")]),
        )
        assert e.agent_type == "evaluator"
        assert isinstance(e, Agent)

    def test_rejects_agent_type_kwarg(self):
        with pytest.raises(TypeError):
            Evaluator(
                name="e",
                instructions="x",
                provider=MockProvider([Message(role="assistant", content="ok")]),
                agent_type="agent",  # type: ignore[call-arg]
            )


class TestFallback:
    def test_type_is_fallback(self):
        f = Fallback(
            name="f",
            instructions="fallback handler",
            provider=MockProvider([Message(role="assistant", content="ok")]),
        )
        assert f.agent_type == "fallback"
        assert isinstance(f, Agent)

    def test_rejects_agent_type_kwarg(self):
        with pytest.raises(TypeError):
            Fallback(
                name="f",
                instructions="x",
                provider=MockProvider([Message(role="assistant", content="ok")]),
                agent_type="agent",  # type: ignore[call-arg]
            )


class TestSubclassDefaults:
    """All subclasses should work with no name or provider (just instructions)."""

    def _provider(self) -> MockProvider:
        return MockProvider([Message(role="assistant", content="ok")])

    def test_guardrail_auto_name(self):
        g = Guardrail(instructions="block bad stuff", provider=self._provider())
        assert g.name.startswith("guardrail-")

    def test_router_auto_name(self):
        r = Router(instructions="route stuff", provider=self._provider())
        assert r.name.startswith("router-")

    def test_evaluator_auto_name(self):
        e = Evaluator(instructions="evaluate", provider=self._provider())
        assert e.name.startswith("evaluator-")

    def test_fallback_auto_name(self):
        f = Fallback(instructions="fallback", provider=self._provider())
        assert f.name.startswith("fallback-")

    def test_guardrail_has_default_instructions(self):
        g = Guardrail(provider=self._provider())
        assert g.instructions

    def test_router_has_default_instructions(self):
        r = Router(provider=self._provider())
        assert r.instructions

    def test_evaluator_has_default_instructions(self):
        e = Evaluator(provider=self._provider())
        assert e.instructions

    def test_fallback_has_default_instructions(self):
        f = Fallback(provider=self._provider())
        assert f.instructions


class TestSubclassesAcceptTools:
    def test_guardrail_with_tools(self, greet_tool):
        g = Guardrail(
            name="g",
            instructions="x",
            provider=MockProvider([Message(role="assistant", content="ok")]),
            tools=[greet_tool],
        )
        assert "greet" in g.tools

    def test_router_with_tools(self, greet_tool):
        r = Router(
            name="r",
            instructions="x",
            provider=MockProvider([Message(role="assistant", content="ok")]),
            tools=[greet_tool],
        )
        assert "greet" in r.tools
