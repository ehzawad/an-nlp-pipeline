"""Context augmentation for fractional queries."""

import json
from typing import Optional, Dict, Tuple
import logging
from .fraction_classifier import FractionalQueryClassifier

logger = logging.getLogger(__name__)


class ContextAugmenter:
    """
    Augments fractional (incomplete) queries with context from previous turns.
    
    Uses ML-based fractional query detection and tag-to-context mappings.
    """

    def __init__(
        self,
        classifier: FractionalQueryClassifier,
        mappings_file: str,
        context_window_turns: int = 1,
        min_confidence: float = 0.7
    ):
        """Initialize context augmenter."""
        self.classifier = classifier
        self.context_window = context_window_turns
        self.min_confidence = min_confidence
        self.mappings = self._load_mappings(mappings_file)

        logger.info(
            f"ContextAugmenter initialized: "
            f"window={context_window_turns}, "
            f"min_confidence={min_confidence}, "
            f"mappings={len(self.mappings)}"
        )

    def _load_mappings(self, filepath: str) -> Dict[str, str]:
        """Load tag-to-context phrase mappings from JSON."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                mappings = json.load(f)

            logger.info(f"Loaded {len(mappings)} tag-to-context mappings from {filepath}")
            return mappings

        except FileNotFoundError:
            logger.warning(
                f"Mappings file not found: {filepath}. "
                "Context augmentation will not work until file is created."
            )
            return {}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse mappings JSON: {e}")
            return {}

    async def augment_if_needed(
        self,
        query: str,
        session_state
    ) -> Tuple[str, dict]:
        """Augment query if it's fractional and context is available."""
        # Check if query is fractional
        is_fractional = await self.classifier.is_fractional(query, self.min_confidence)

        if not is_fractional:
            return query, {}

        # Query is fractional, try to get context
        confidence = await self.classifier.predict_proba(query)

        logger.info(
            f"Fractional query detected (confidence={confidence:.3f}): '{query}'"
        )

        # Get context tag from previous turn(s)
        context_tag = self._get_last_context_tag(session_state)

        if not context_tag:
            logger.info("No context tag available in session history")
            return query, {
                "fractional_detected": True,
                "fractional_confidence": confidence,
                "no_context": True
            }

        # Map tag to context phrase
        context_phrase = self.mappings.get(context_tag)

        if not context_phrase:
            logger.info(
                f"No mapping found for tag '{context_tag}'. "
                "Add to context_mappings.json to enable augmentation."
            )
            return query, {
                "fractional_detected": True,
                "fractional_confidence": confidence,
                "context_tag": context_tag,
                "no_mapping": True
            }

        # Augment query
        augmented_query = self._augment_query(query, context_phrase)

        logger.info(
            f"Augmented query: '{query}' → '{augmented_query}' "
            f"(context_tag='{context_tag}')"
        )

        metadata = {
            "augmented": True,
            "original_query": query,
            "augmented_query": augmented_query,
            "context_tag": context_tag,
            "context_phrase": context_phrase,
            "fractional_confidence": confidence
        }

        return augmented_query, metadata

    def _get_last_context_tag(self, session_state) -> Optional[str]:
        """Get context tag from last N conversation turns."""
        # Check if session has last_context_tag field
        if hasattr(session_state, 'last_context_tag'):
            return session_state.last_context_tag

        # Fallback: check conversation history
        if hasattr(session_state, 'conversation_history') and session_state.conversation_history:
            history = session_state.conversation_history

            # Look through last N turns
            for i in range(min(self.context_window, len(history))):
                turn_idx = -(i + 1)
                turn = history[turn_idx]

                # Check if turn has tag info
                if isinstance(turn, dict):
                    tag = turn.get('tag') or turn.get('context_tag')
                    if tag:
                        return tag

        return None

    def _augment_query(self, query: str, context_phrase: str) -> str:
        """Prepend context phrase to query."""
        return f"{context_phrase} {query}"

    def reload_mappings(self, filepath: str):
        """Reload tag-to-context mappings from file."""
        self.mappings = self._load_mappings(filepath)
        logger.info("Context mappings reloaded")

