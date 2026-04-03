"""Calendar tools — in-memory event management."""

from edge_agent import tool

_events: list[dict[str, str]] = [
    {"date": "2026-03-31", "time": "10:00", "title": "Team standup"},
    {"date": "2026-03-31", "time": "14:00", "title": "Dentist appointment"},
    {"date": "2026-04-01", "time": "09:00", "title": "Sprint planning"},
    {"date": "2026-04-02", "time": "11:00", "title": "Coffee with Sarah"},
]


@tool
def get_events(date: str) -> str:
    """Get all events for a specific date (YYYY-MM-DD format)."""
    matching = [e for e in _events if e["date"] == date]
    if not matching:
        return f"No events on {date}."
    lines = [f"  {e['time']} — {e['title']}" for e in matching]
    return f"Events for {date}:\n" + "\n".join(lines)


@tool
def add_event(date: str, time: str, title: str) -> str:
    """Add a new calendar event."""
    _events.append({"date": date, "time": time, "title": title})
    return f"Event '{title}' added on {date} at {time}."


@tool
def cancel_event(title: str) -> str:
    """Cancel (delete) an event by its title."""
    for i, e in enumerate(_events):
        if e["title"].lower() == title.lower():
            _events.pop(i)
            return f"Event '{title}' cancelled."
    return f"No event found with title '{title}'."


@tool
def list_upcoming_events() -> str:
    """List all upcoming events across all dates."""
    if not _events:
        return "No upcoming events."
    sorted_events = sorted(_events, key=lambda e: (e["date"], e["time"]))
    lines = [f"  {e['date']} {e['time']} — {e['title']}" for e in sorted_events]
    return "Upcoming events:\n" + "\n".join(lines)


ALL_CALENDAR_TOOLS = [get_events, add_event, cancel_event, list_upcoming_events]
