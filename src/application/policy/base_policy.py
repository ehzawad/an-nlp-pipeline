"""Base policy interface and action types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class ActionType(Enum):
    """Types of actions the dialogue system can take."""
    FAQ_ANSWER = "faq_answer"
    CLARIFY_INTENT = "clarify_intent"
    COLLECT_SLOT = "collect_slot"
    START_FORM = "start_form"
    EXECUTE_FORM = "execute_form"
    PAUSE_FORM = "pause_form"
    RESUME_FORM = "resume_form"
    ESCALATE = "escalate"
    GREETING = "greeting"
    FALLBACK = "fallback"
    SWITCH_INTENT = "switch_intent"
    ASK_INTERRUPTION_CHOICE = "ask_interruption_choice"


@dataclass
class Action:
    """Represents an action to be executed by the dialogue manager."""
    action_type: ActionType
    data: Dict[str, Any]
    response_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def faq_answer(cls, results: List[dict], cluster: Optional[str] = None) -> "Action":
        """Create FAQ answer action."""
        return cls(
            action_type=ActionType.FAQ_ANSWER,
            data={"results": results, "cluster": cluster}
        )

    @classmethod
    def clarify_intent(cls, top_options: List[dict], confidence: float) -> "Action":
        """Create clarification action."""
        return cls(
            action_type=ActionType.CLARIFY_INTENT,
            data={"top_options": top_options, "confidence": confidence}
        )

    @classmethod
    def escalate(cls, reason: str) -> "Action":
        """Create escalation action."""
        return cls(
            action_type=ActionType.ESCALATE,
            data={"reason": reason}
        )

    @classmethod
    def fallback(cls) -> "Action":
        """Create fallback action."""
        return cls(
            action_type=ActionType.FALLBACK,
            data={}
        )


class BasePolicy(ABC):
    """Abstract base class for dialogue policies."""

    @abstractmethod
    async def decide(
        self,
        session_state: Any,
        user_query: str,
        nlp_result: dict,
        entities: Optional[Dict[str, Any]] = None
    ) -> Action:
        """Decide next action based on inputs."""
        pass

    def _get_confidence(self, nlp_result: dict) -> float:
        """Extract confidence from NLP result."""
        if "confidence" in nlp_result:
            return nlp_result["confidence"]
        elif "classification" in nlp_result and nlp_result["classification"]:
            return nlp_result["classification"][0].get("confidence", 0.0)
        return 0.0

    def _get_top_cluster(self, nlp_result: dict) -> Optional[str]:
        """Extract top cluster from NLP result."""
        if "classification" in nlp_result and nlp_result["classification"]:
            return nlp_result["classification"][0].get("cluster")
        return None

    def _get_top_k_clusters(self, nlp_result: dict, k: int = 3) -> List[dict]:
        """Extract top-k clusters from NLP result."""
        if "classification" in nlp_result:
            return nlp_result["classification"][:k]
        return []

