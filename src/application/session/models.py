"""Session state models for dialogue management."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TaskType(Enum):
    """Types of tasks the dialogue system can handle."""
    FAQ = "faq"
    FORM_FILLING = "form_filling"
    TROUBLESHOOTING = "troubleshooting"
    STATUS_CHECK = "status_check"


@dataclass
class DialogueTurn:
    """Single turn in a conversation."""
    timestamp: datetime
    user_query: str
    bot_response: str
    nlp_result: Optional[dict] = None
    action_taken: Optional[str] = None
    entities_extracted: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Serialize turn to dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_query": self.user_query,
            "bot_response": self.bot_response,
            "nlp_result": self.nlp_result,
            "action_taken": self.action_taken,
            "entities_extracted": self.entities_extracted
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DialogueTurn":
        """Deserialize turn from dict."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_query=data["user_query"],
            bot_response=data["bot_response"],
            nlp_result=data.get("nlp_result"),
            action_taken=data.get("action_taken"),
            entities_extracted=data.get("entities_extracted")
        )


@dataclass
class FormStateData:
    """State of a partially-filled form."""
    form_name: str
    current_slot: Optional[str] = None
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    slot_retries: Dict[str, int] = field(default_factory=dict)
    interruption_count: int = 0
    paused_at: Optional[datetime] = None
    awaiting_interruption_response: bool = False

    def to_dict(self) -> dict:
        """Serialize form state."""
        return {
            "form_name": self.form_name,
            "current_slot": self.current_slot,
            "filled_slots": self.filled_slots,
            "slot_retries": self.slot_retries,
            "interruption_count": self.interruption_count,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "awaiting_interruption_response": self.awaiting_interruption_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FormStateData":
        """Deserialize form state."""
        return cls(
            form_name=data["form_name"],
            current_slot=data.get("current_slot"),
            filled_slots=data.get("filled_slots", {}),
            slot_retries=data.get("slot_retries", {}),
            interruption_count=data.get("interruption_count", 0),
            paused_at=datetime.fromisoformat(data["paused_at"]) if data.get("paused_at") else None,
            awaiting_interruption_response=data.get("awaiting_interruption_response", False),
        )


@dataclass
class SessionState:
    """Complete session state for a conversation."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Conversation history (circular buffer)
    conversation_history: List[DialogueTurn] = field(default_factory=list)
    max_history: int = 10

    # Current task
    active_task: Optional[TaskType] = None
    active_form: Optional[FormStateData] = None

    # Context
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    last_context_tag: Optional[str] = None

    # Error tracking
    error_count: int = 0
    max_errors: int = 3
    escalation_flag: bool = False

    def add_turn(self, user_query: str, bot_response: str, **kwargs):
        """Add a turn to conversation history (circular buffer)."""
        turn = DialogueTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            bot_response=bot_response,
            **kwargs
        )
        self.conversation_history.append(turn)

        # Keep only last N turns
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        self.last_activity = datetime.now()

    def get_last_n_turns(self, n: int = 5) -> List[DialogueTurn]:
        """Get last N conversation turns."""
        return self.conversation_history[-n:] if self.conversation_history else []

    def increment_error(self):
        """Increment error count."""
        self.error_count += 1
        if self.error_count >= self.max_errors:
            self.escalation_flag = True

    def reset_errors(self):
        """Reset error count."""
        self.error_count = 0

    def in_form_state(self) -> bool:
        """Check if actively in a form."""
        return self.active_form is not None

    def to_dict(self) -> dict:
        """Serialize session state."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "max_history": self.max_history,
            "active_task": self.active_task.value if self.active_task else None,
            "active_form": self.active_form.to_dict() if self.active_form else None,
            "extracted_entities": self.extracted_entities,
            "last_context_tag": self.last_context_tag,
            "error_count": self.error_count,
            "max_errors": self.max_errors,
            "escalation_flag": self.escalation_flag
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        """Deserialize session state."""
        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            max_history=data.get("max_history", 10),
            extracted_entities=data.get("extracted_entities", {}),
            last_context_tag=data.get("last_context_tag"),
            error_count=data.get("error_count", 0),
            max_errors=data.get("max_errors", 3),
            escalation_flag=data.get("escalation_flag", False)
        )

        # Deserialize conversation history
        if "conversation_history" in data:
            session.conversation_history = [
                DialogueTurn.from_dict(turn_data)
                for turn_data in data["conversation_history"]
            ]

        # Deserialize task
        if data.get("active_task"):
            session.active_task = TaskType(data["active_task"])

        # Deserialize form
        if data.get("active_form"):
            session.active_form = FormStateData.from_dict(data["active_form"])

        return session

