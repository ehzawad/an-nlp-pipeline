"""FAQ policy for direct question answering."""

from typing import Optional, Dict, Any
import logging
from .base_policy import BasePolicy, Action

logger = logging.getLogger(__name__)


class FAQPolicy(BasePolicy):
    """
    Policy for FAQ retrieval.
    
    Returns direct answers when:
    - High confidence classification
    - Good search results available
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize FAQ policy."""
        self.confidence_threshold = confidence_threshold

    async def decide(
        self,
        session_state,
        user_query: str,
        nlp_result: dict,
        entities: Optional[Dict[str, Any]] = None
    ) -> Action:
        """Decide if query should be answered directly."""
        confidence = self._get_confidence(nlp_result)
        cluster = self._get_top_cluster(nlp_result)
        results = (
            nlp_result.get("results")
            or nlp_result.get("search_results")
            or []
        )

        logger.debug(
            f"FAQ Policy: confidence={confidence:.3f}, "
            f"cluster={cluster}, results={len(results)}"
        )

        # Check if we have good results
        if confidence >= self.confidence_threshold and results:
            logger.info(
                f"FAQ Policy → Answer directly (cluster={cluster}, "
                f"confidence={confidence:.3f})"
            )
            return Action.faq_answer(results=results, cluster=cluster)

        else:
            # Not confident enough or no results, need clarification
            logger.info(
                f"FAQ Policy → Need clarification (confidence={confidence:.3f} "
                f"< {self.confidence_threshold})"
            )
            return Action.clarify_intent(
                top_options=self._get_top_k_clusters(nlp_result, k=3),
                confidence=confidence
            )

