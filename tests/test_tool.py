import dataclasses
import enum
from typing import Literal, Optional

from edge_agent import tool
from edge_agent.tool import Tool


class TestToolDecorator:
    def test_produces_tool_instance(self):
        @tool
        def noop() -> str:
            """Do nothing."""
            return ""

        assert isinstance(noop, Tool)

    def test_name_from_function(self):
        @tool
        def my_func() -> str:
            """A function."""
            return ""

        assert my_func.name == "my_func"

    def test_description_from_docstring(self):
        @tool
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}"

        assert greet.description == "Greet someone by name."

    def test_empty_docstring(self):
        @tool
        def nodoc(x: str) -> str:
            return x

        assert nodoc.description == ""

    def test_schema_required_params(self):
        @tool
        def search(query: str, limit: int) -> str:
            """Search for things."""
            return ""

        assert search.parameters == {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query", "limit"],
        }

    def test_schema_optional_params(self):
        @tool
        def greet(name: str, excited: bool = False) -> str:
            """Greet someone."""
            return ""

        params = greet.parameters
        assert "required" in params
        assert params["required"] == ["name"]
        assert "excited" not in params["required"]

    def test_schema_no_required(self):
        @tool
        def defaults_only(x: str = "a", y: int = 0) -> str:
            """All optional."""
            return ""

        assert "required" not in defaults_only.parameters

    def test_type_mapping_str(self):
        @tool
        def f(x: str) -> str:
            """Test."""
            return x

        assert f.parameters["properties"]["x"]["type"] == "string"

    def test_type_mapping_int(self):
        @tool
        def f(x: int) -> str:
            """Test."""
            return str(x)

        assert f.parameters["properties"]["x"]["type"] == "integer"

    def test_type_mapping_float(self):
        @tool
        def f(x: float) -> str:
            """Test."""
            return str(x)

        assert f.parameters["properties"]["x"]["type"] == "number"

    def test_type_mapping_bool(self):
        @tool
        def f(x: bool) -> str:
            """Test."""
            return str(x)

        assert f.parameters["properties"]["x"]["type"] == "boolean"

    def test_type_mapping_list(self):
        @tool
        def f(x: list) -> str:
            """Test."""
            return str(x)

        assert f.parameters["properties"]["x"]["type"] == "array"

    def test_type_mapping_list_of_str(self):
        @tool
        def f(x: list[str]) -> str:
            """Test."""
            return str(x)

        prop = f.parameters["properties"]["x"]
        assert prop == {"type": "array", "items": {"type": "string"}}

    def test_type_mapping_list_of_int(self):
        @tool
        def f(x: list[int]) -> str:
            """Test."""
            return str(x)

        prop = f.parameters["properties"]["x"]
        assert prop == {"type": "array", "items": {"type": "integer"}}

    def test_type_mapping_optional_str(self):
        @tool
        def f(x: Optional[str] = None) -> str:
            """Test."""
            return str(x)

        prop = f.parameters["properties"]["x"]
        assert prop == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    def test_type_mapping_union_none(self):
        @tool
        def f(x: str | None = None) -> str:
            """Test."""
            return str(x)

        prop = f.parameters["properties"]["x"]
        assert prop == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    def test_type_mapping_literal_strings(self):
        @tool
        def f(mode: Literal["fast", "slow"]) -> str:
            """Test."""
            return mode

        prop = f.parameters["properties"]["mode"]
        assert prop == {"type": "string", "enum": ["fast", "slow"]}

    def test_type_mapping_literal_ints(self):
        @tool
        def f(level: Literal[1, 2, 3]) -> str:
            """Test."""
            return str(level)

        prop = f.parameters["properties"]["level"]
        assert prop == {"type": "integer", "enum": [1, 2, 3]}

    def test_type_mapping_enum(self):
        class Color(enum.Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        @tool
        def f(color: Color) -> str:
            """Test."""
            return color.value

        prop = f.parameters["properties"]["color"]
        assert prop == {"type": "string", "enum": ["red", "green", "blue"]}

    def test_type_mapping_nested_dataclass(self):
        @dataclasses.dataclass
        class Address:
            street: str
            city: str

        @tool
        def f(addr: Address) -> str:
            """Test."""
            return addr.city

        prop = f.parameters["properties"]["addr"]
        assert prop["type"] == "object"
        assert "street" in prop["properties"]
        assert "city" in prop["properties"]
        assert set(prop["required"]) == {"street", "city"}

    def test_type_mapping_list_of_dataclass(self):
        @dataclasses.dataclass
        class Item:
            name: str

        @tool
        def f(items: list[Item]) -> str:
            """Test."""
            return ""

        prop = f.parameters["properties"]["items"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "object"
        assert "name" in prop["items"]["properties"]

    def test_decorated_function_is_callable(self):
        @tool
        def add(a: int, b: int) -> str:
            """Add numbers."""
            return str(a + b)

        assert add(a=1, b=2) == "3"

    def test_full_schema_example(self):
        @tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            """Get the current weather for a city."""
            return f"22 degrees {unit} in {city}"

        assert get_weather.name == "get_weather"
        assert get_weather.description == "Get the current weather for a city."
        assert get_weather.parameters == {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["city"],
        }
        assert get_weather(city="Tokyo") == "22 degrees celsius in Tokyo"
