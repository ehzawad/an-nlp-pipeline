"""Passthrough summarizer that returns input unchanged."""

from typing import Tuple, Dict, Any
from .base_summarizer import BaseSummarizer


class PassthroughSummarizer(BaseSummarizer):
    """
    Identity function summarizer - returns input unchanged.
    
    This is a placeholder implementation used for:
    - Testing the summarization pipeline without actual summarization
    - Gradual rollout (infrastructure ready, summarization comes later)
    - Environments where summarization is not needed
    """

    def __init__(self):
        """Initialize passthrough summarizer."""
        pass

    async def summarize(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Return input text unchanged."""
        metadata = {
            "summarized": False,
            "method": "passthrough",
            "reason": "passthrough_implementation"
        }

        return text, metadata

