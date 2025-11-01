"""Base summarizer interface."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseSummarizer(ABC):
    """Abstract base class for query summarizers."""

    @abstractmethod
    async def summarize(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Summarize input text.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Tuple of (summarized_text, metadata)
        """
        pass

