"""Aggregated tool registry — single import point for all domain tools."""

from .math_tools import ALL_MATH_TOOLS
from .weather_tools import ALL_WEATHER_TOOLS
from .notes_tools import ALL_NOTES_TOOLS
from .calendar_tools import ALL_CALENDAR_TOOLS

ALL_TOOLS = ALL_MATH_TOOLS + ALL_WEATHER_TOOLS + ALL_NOTES_TOOLS + ALL_CALENDAR_TOOLS

TOOLS_BY_DOMAIN = {
    "math": ALL_MATH_TOOLS,
    "weather": ALL_WEATHER_TOOLS,
    "notes": ALL_NOTES_TOOLS,
    "calendar": ALL_CALENDAR_TOOLS,
}
