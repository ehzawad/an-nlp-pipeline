"""
Dialogue Manager Component - Central Conversation Orchestrator

Manages multi-turn dialogues by:
- Text fragmentation and fractionation detection
- Routing to appropriate NLP components
- NER extraction and entity tracking
- Action invocation (forms, APIs, etc.)
- Session state management
- Message history tracking
- Context-aware processing

This is the main entry point for dialogue systems.
"""

from src.components.dialogue.manager import (
    DialogueManager,
    DialogueInput,
    DialogueOutput,
    DialogueConfig,
    SessionState,
    Message,
    MessageRole,
)

__all__ = [
    "DialogueManager",
    "DialogueInput",
    "DialogueOutput",
    "DialogueConfig",
    "SessionState",
    "Message",
    "MessageRole",
]
