import pytest

from edge_agent.providers.base import Provider
from edge_agent.tool import Tool
from edge_agent.types import Message


class TestProviderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Provider()

    def test_subclass_without_chat_raises(self):
        class Incomplete(Provider):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_with_chat_can_instantiate(self):
        class Complete(Provider):
            def chat(
                self,
                messages: list[Message],
                tools: list[Tool] | None = None,
                output_schema: dict[str, object] | None = None,
            ) -> Message:
                return Message(role="assistant", content="ok")

        provider = Complete()
        result = provider.chat([Message(role="user", content="hi")])
        assert result.content == "ok"
