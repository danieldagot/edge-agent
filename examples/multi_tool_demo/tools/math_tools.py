"""Math tools — arithmetic operations."""

from edge_agent import tool


@tool
def add(a: int, b: int) -> str:
    """Add two numbers and return the result."""
    return str(a + b)


@tool
def subtract(a: int, b: int) -> str:
    """Subtract b from a and return the result."""
    return str(a - b)


@tool
def multiply(a: int, b: int) -> str:
    """Multiply two numbers and return the result."""
    return str(a * b)


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b and return the result."""
    if b == 0:
        return "Error: division by zero"
    return str(a / b)


ALL_MATH_TOOLS = [add, subtract, multiply, divide]
