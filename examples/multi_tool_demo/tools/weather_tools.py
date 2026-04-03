"""Weather tools — forecasts and conditions."""

from edge_agent import tool

_CURRENT = {
    "london": "14°C, cloudy with a chance of rain",
    "new york": "22°C, sunny",
    "tokyo": "18°C, partly cloudy",
    "sydney": "25°C, clear skies",
    "paris": "16°C, overcast",
    "berlin": "12°C, light drizzle",
}

_FORECAST = {
    "london": "Tomorrow: 16°C, sunny. Wednesday: 13°C, showers.",
    "new york": "Tomorrow: 24°C, partly cloudy. Wednesday: 20°C, thunderstorms.",
    "tokyo": "Tomorrow: 20°C, clear. Wednesday: 19°C, rain.",
    "sydney": "Tomorrow: 27°C, sunny. Wednesday: 23°C, windy.",
    "paris": "Tomorrow: 18°C, clear. Wednesday: 15°C, cloudy.",
    "berlin": "Tomorrow: 14°C, sunny. Wednesday: 11°C, rain.",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather conditions for a city."""
    return _CURRENT.get(city.lower(), f"No weather data for {city}")


@tool
def get_forecast(city: str) -> str:
    """Get the weather forecast for the next few days."""
    return _FORECAST.get(city.lower(), f"No forecast data for {city}")


@tool
def compare_weather(city_a: str, city_b: str) -> str:
    """Compare the current weather between two cities."""
    a = _CURRENT.get(city_a.lower(), "unknown")
    b = _CURRENT.get(city_b.lower(), "unknown")
    return f"{city_a}: {a} | {city_b}: {b}"


ALL_WEATHER_TOOLS = [get_weather, get_forecast, compare_weather]
