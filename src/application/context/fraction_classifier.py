"""Fractional query classifier for detecting incomplete queries."""

import pickle
import numpy as np
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class FractionalQueryClassifier:
    """
    Binary classifier to detect fractional (incomplete) queries.
    
    Uses pre-trained logistic regression on E5 embeddings.
    """

    def __init__(
        self,
        model_path: str,
        embedding_model: Any
    ):
        """Initialize fractional query classifier."""
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.classifier = self._load_classifier(model_path)

        logger.info(
            f"FractionalQueryClassifier initialized with model: {model_path}"
        )

    def _load_classifier(self, path: str):
        """Load pickled classifier model."""
        with open(path, 'rb') as f:
            classifier = pickle.load(f)

        logger.info(f"Loaded classifier from: {path}")
        return classifier

    async def _encode_query(self, query: str) -> np.ndarray:
        """Encode query using E5 model (async-compatible)."""
        # E5 requires "query: " prefix
        query_with_prefix = f"query: {query}"

        # This will be called in thread pool by the caller
        embedding = self.embedding_model.encode(
            [query_with_prefix],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding[0]

    async def is_fractional(self, query: str, threshold: float = 0.7) -> bool:
        """Detect if query is fractional (incomplete)."""
        probability = await self.predict_proba(query)
        return probability >= threshold

    async def predict_proba(self, query: str) -> float:
        """Get probability that query is fractional."""
        # Encode query
        embedding = await self._encode_query(query)

        # Predict probability
        proba = self.classifier.predict_proba([embedding])[0]

        # Return probability of fractional class (index 1)
        return float(proba[1])

    async def predict(self, query: str) -> int:
        """Predict class label (0=complete, 1=fractional)."""
        embedding = await self._encode_query(query)
        prediction = self.classifier.predict([embedding])[0]
        return int(prediction)


# Import at module level for type hints
from typing import Any

