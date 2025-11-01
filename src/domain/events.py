"""
Domain events - Immutable event log for event sourcing.

Every state change is captured as an event, enabling:
- Complete audit trail
- Time-travel debugging
- Event replay
- Analytics
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum


class EventType(str, Enum):
    """Type-safe event types (NO MAGIC STRINGS!)"""
    
    # Session events
    SESSION_CREATED = "session.created"
    SESSION_DESTROYED = "session.destroyed"
    
    # Turn events
    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    TURN_FAILED = "turn.failed"
    
    # NLP events
    NLP_STARTED = "nlp.started"
    NLP_COMPLETED = "nlp.completed"
    NLP_FAILED = "nlp.failed"
    
    # Policy events
    POLICY_DECIDED = "policy.decided"
    
    # Form events
    FORM_STARTED = "form.started"
    FORM_SLOT_REQUESTED = "form.slot_requested"
    FORM_SLOT_FILLED = "form.slot_filled"
    FORM_SLOT_INVALID = "form.slot_invalid"
    FORM_COMPLETED = "form.completed"
    FORM_FAILED = "form.failed"
    FORM_PAUSED = "form.paused"
    FORM_RESUMED = "form.resumed"
    
    # Error events
    ERROR_OCCURRED = "error.occurred"
    ESCALATION_TRIGGERED = "escalation.triggered"


@dataclass(frozen=True)  # Immutable!
class DomainEvent:
    """Base domain event - immutable by design"""
    
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: EventType = field(default=EventType.TURN_STARTED)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    aggregate_id: str = field(default="")  # Session ID
    tenant_id: str = field(default="")
    user_id: str = field(default="")
    
    # Event payload
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata for tracing
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None  # ID of event that caused this
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for event store"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "aggregate_id": self.aggregate_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainEvent":
        """Deserialize from event store"""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            aggregate_id=data["aggregate_id"],
            tenant_id=data["tenant_id"],
            user_id=data["user_id"],
            data=data["data"],
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
        )


# ============================================================================
# Event Factory Functions (Type-Safe Event Creation)
# ============================================================================

def session_created(
    session_id: str,
    tenant_id: str,
    user_id: str,
    correlation_id: Optional[str] = None
) -> DomainEvent:
    """Factory for session creation event"""
    return DomainEvent(
        event_type=EventType.SESSION_CREATED,
        aggregate_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        data={"session_id": session_id},
        correlation_id=correlation_id,
    )


def turn_started(
    session_id: str,
    tenant_id: str,
    user_id: str,
    query: str,
    correlation_id: str,
) -> DomainEvent:
    """Factory for turn start event"""
    return DomainEvent(
        event_type=EventType.TURN_STARTED,
        aggregate_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        data={"query": query, "query_length": len(query)},
        correlation_id=correlation_id,
    )


def nlp_completed(
    session_id: str,
    tenant_id: str,
    user_id: str,
    nlp_result: Dict[str, Any],
    correlation_id: str,
    causation_id: str,
) -> DomainEvent:
    """Factory for NLP completion event"""
    return DomainEvent(
        event_type=EventType.NLP_COMPLETED,
        aggregate_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        data={"nlp_result": nlp_result},
        correlation_id=correlation_id,
        causation_id=causation_id,
    )


def form_slot_filled(
    session_id: str,
    tenant_id: str,
    user_id: str,
    form_name: str,
    slot_name: str,
    slot_value: Any,
    correlation_id: str,
    causation_id: str,
) -> DomainEvent:
    """Factory for form slot filled event"""
    return DomainEvent(
        event_type=EventType.FORM_SLOT_FILLED,
        aggregate_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        data={
            "form_name": form_name,
            "slot_name": slot_name,
            "slot_value": slot_value,
        },
        correlation_id=correlation_id,
        causation_id=causation_id,
    )


def error_occurred(
    session_id: str,
    tenant_id: str,
    user_id: str,
    error_type: str,
    error_message: str,
    correlation_id: str,
    causation_id: Optional[str] = None,
) -> DomainEvent:
    """Factory for error event"""
    return DomainEvent(
        event_type=EventType.ERROR_OCCURRED,
        aggregate_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        data={
            "error_type": error_type,
            "error_message": error_message,
        },
        correlation_id=correlation_id,
        causation_id=causation_id,
    )
