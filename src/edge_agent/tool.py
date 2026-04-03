from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(py_type: type) -> str:
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        py_type = origin
    return _PYTHON_TYPE_TO_JSON.get(py_type, "string")


class Tool:
    """Wraps a plain function with its JSON-schema metadata so providers can
    advertise it to the LLM and the agent loop can execute it."""

    __slots__ = ("fn", "name", "description", "parameters")

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description
        self.parameters = parameters

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


def _build_parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    hints.pop("return", None)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        py_type = hints.get(param_name, str)
        prop: dict[str, str] = {"type": _json_type(py_type)}
        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def tool(fn: Callable[..., Any]) -> Tool:
    """Decorator that turns a typed function into a :class:`Tool`.

    The function's name, docstring, and type hints are used to build the
    JSON-schema description that LLM providers need for function calling.
    """
    name = fn.__name__
    description = (fn.__doc__ or "").strip()
    parameters = _build_parameters_schema(fn)
    return Tool(fn=fn, name=name, description=description, parameters=parameters)
