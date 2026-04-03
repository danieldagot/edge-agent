"""Notes tools — in-memory note management."""

from edge_agent import tool

_notes: dict[str, str] = {}


@tool
def add_note(title: str, content: str) -> str:
    """Create a new note with a title and content."""
    _notes[title] = content
    return f"Note '{title}' saved."


@tool
def get_note(title: str) -> str:
    """Retrieve a note by its title."""
    if title in _notes:
        return f"{title}: {_notes[title]}"
    return f"No note found with title '{title}'"


@tool
def list_notes() -> str:
    """List all saved note titles."""
    if not _notes:
        return "No notes saved yet."
    return "Notes: " + ", ".join(_notes.keys())


@tool
def delete_note(title: str) -> str:
    """Delete a note by its title."""
    if title in _notes:
        del _notes[title]
        return f"Note '{title}' deleted."
    return f"No note found with title '{title}'"


ALL_NOTES_TOOLS = [add_note, get_note, list_notes, delete_note]
