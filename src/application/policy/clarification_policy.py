"""Clarification policy for ambiguous queries."""

from typing import Optional, Dict, Any
import logging
from .base_policy import BasePolicy, Action

logger = logging.getLogger(__name__)


class ClarificationPolicy(BasePolicy):
    """
    Policy for handling ambiguous queries.
    
    When confidence is medium (not high enough for direct answer,
    but not low enough to escalate), ask user to clarify.
    """

    def __init__(self, min_confidence: float = 0.5):
        """Initialize clarification policy."""
        self.min_confidence = min_confidence

    async def decide(
        self,
        session_state,
        user_query: str,
        nlp_result: dict,
        entities: Optional[Dict[str, Any]] = None
    ) -> Action:
        """Generate clarification question."""
        confidence = self._get_confidence(nlp_result)
        top_options = self._get_top_k_clusters(nlp_result, k=3)

        if confidence >= self.min_confidence and top_options:
            logger.info(
                f"Clarification Policy → Ask user to choose (confidence={confidence:.3f})"
            )
            return Action.clarify_intent(
                top_options=top_options,
                confidence=confidence
            )
        else:
            # Too low confidence, escalate
            logger.info(
                f"Clarification Policy → Confidence too low ({confidence:.3f}), escalating"
            )
            return Action.escalate(
                reason=f"Low confidence: {confidence:.3f}"
            )

