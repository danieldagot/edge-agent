from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from edge_agent.schema import parse_dataclass, schema_from_dataclass


@dataclass
class Flat:
    name: str
    age: int
    score: float
    active: bool


@dataclass
class Inner:
    label: str
    value: int


@dataclass
class Nested:
    title: str
    inner: Inner


@dataclass
class WithDefaults:
    required_field: str
    optional_str: str = "default"
    optional_int: int = 0


@dataclass
class WithList:
    tags: list[str]
    items: list[Inner]


@dataclass
class AllOptional:
    x: str = "a"
    y: int = 0


class TestSchemaFromDataclass:
    def test_flat_dataclass_schema(self):
        schema = schema_from_dataclass(Flat)
        assert schema["type"] == "object"

        props = schema["properties"]
        assert props["name"] == {"type": "string"}
        assert props["age"] == {"type": "integer"}
        assert props["score"] == {"type": "number"}
        assert props["active"] == {"type": "boolean"}

        assert set(schema["required"]) == {"name", "age", "score", "active"}

    def test_nested_dataclass_schema(self):
        schema = schema_from_dataclass(Nested)
        props = schema["properties"]
        assert props["title"] == {"type": "string"}

        inner_schema = props["inner"]
        assert inner_schema["type"] == "object"
        assert inner_schema["properties"]["label"] == {"type": "string"}
        assert inner_schema["properties"]["value"] == {"type": "integer"}

    def test_optional_fields_excluded_from_required(self):
        schema = schema_from_dataclass(WithDefaults)
        assert schema["required"] == ["required_field"]

    def test_all_optional_has_no_required(self):
        schema = schema_from_dataclass(AllOptional)
        assert "required" not in schema

    def test_list_field_schema(self):
        schema = schema_from_dataclass(WithList)
        props = schema["properties"]

        assert props["tags"]["type"] == "array"
        assert props["tags"]["items"] == {"type": "string"}

        assert props["items"]["type"] == "array"
        items_schema = props["items"]["items"]
        assert items_schema["type"] == "object"
        assert items_schema["properties"]["label"] == {"type": "string"}

    def test_non_dataclass_raises(self):
        with pytest.raises(TypeError, match="is not a dataclass"):
            schema_from_dataclass(str)


class TestParseDataclass:
    def test_parse_flat(self):
        data: dict[str, object] = {
            "name": "Alice",
            "age": 30,
            "score": 9.5,
            "active": True,
        }
        result = parse_dataclass(Flat, data)
        assert isinstance(result, Flat)
        assert result.name == "Alice"
        assert result.age == 30
        assert result.score == 9.5
        assert result.active is True

    def test_parse_nested(self):
        data: dict[str, object] = {
            "title": "Test",
            "inner": {"label": "sub", "value": 42},
        }
        result = parse_dataclass(Nested, data)
        assert isinstance(result, Nested)
        assert isinstance(result.inner, Inner)
        assert result.inner.label == "sub"
        assert result.inner.value == 42

    def test_parse_with_defaults_missing_optional(self):
        data: dict[str, object] = {"required_field": "hello"}
        result = parse_dataclass(WithDefaults, data)
        assert result.required_field == "hello"
        assert result.optional_str == "default"
        assert result.optional_int == 0

    def test_parse_list_of_dataclasses(self):
        data: dict[str, object] = {
            "tags": ["a", "b"],
            "items": [
                {"label": "x", "value": 1},
                {"label": "y", "value": 2},
            ],
        }
        result = parse_dataclass(WithList, data)
        assert result.tags == ["a", "b"]
        assert len(result.items) == 2
        assert isinstance(result.items[0], Inner)
        assert result.items[0].label == "x"

    def test_non_dataclass_raises(self):
        with pytest.raises(TypeError, match="is not a dataclass"):
            parse_dataclass(str, {})
