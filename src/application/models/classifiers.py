"""Classifier model implementations."""

import pickle
import numpy as np
from typing import List, Dict, Any
from .base import BaseModel, ModelConfig
from .embeddings import EmbeddingModel
import logging

logger = logging.getLogger(__name__)


class ClassifierModel(BaseModel):
    """48-class intent classifier."""

    def __init__(self, config: ModelConfig, embedding_model: EmbeddingModel):
        """
        Initialize classifier with embedding model dependency.
        
        Args:
            config: Model configuration
            embedding_model: Embedding model for feature extraction
        """
        super().__init__(config)
        self.embedding_model = embedding_model
        self.classifier = None
        self.label_encoder = None
        self.model_path = config.config.get("model_path", "models/classification/model.pkl")
        self.label_encoder_path = config.config.get(
            "label_encoder_path",
            "models/classification/label_encoder.pkl"
        )

    def load(self) -> None:
        """Load classifier and label encoder."""
        if self._loaded:
            logger.info("Classifier already loaded")
            return

        logger.info(f"Loading classifier from {self.model_path}")

        try:
            # Load classifier
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)

            # Load label encoder
            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

            self._loaded = True
            logger.info(
                f"Classifier loaded: {len(self.label_encoder.classes_)} classes"
            )

        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if classifier is loaded."""
        return self._loaded and self.classifier is not None

    def predict(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Predict top-k classes for query.
        
        Args:
            query: User query
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with cluster and confidence
        """
        if not self.is_loaded():
            self.load()

        # Get embedding
        embedding = self.embedding_model.embed(query)
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        # Get probabilities
        probabilities = self.classifier.predict_proba(embedding)[0]

        # Get top-k indices
        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            cluster = self.label_encoder.classes_[idx]
            confidence = float(probabilities[idx])
            results.append({
                "cluster": cluster,
                "confidence": confidence
            })

        return results

    def predict_single(self, query: str) -> Dict[str, Any]:
        """Predict single top class."""
        results = self.predict(query, top_k=1)
        return results[0] if results else {"cluster": "unknown", "confidence": 0.0}

