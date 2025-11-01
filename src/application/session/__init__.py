"""Session management module."""

from .models import SessionState, DialogueTurn, FormStateData, TaskType
from .store import InMemorySessionStore
from .manager import SessionManager

# Export SessionStore alias for backward compatibility
SessionStore = InMemorySessionStore

__all__ = [
    "SessionState",
    "DialogueTurn",
    "FormStateData",
    "TaskType",
    "InMemorySessionStore",
    "SessionStore",  # Backward compatible alias
    "SessionManager",
]
