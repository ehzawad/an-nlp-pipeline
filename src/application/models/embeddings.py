"""Embedding model implementations."""

from typing import List, Union
import numpy as np
from .base import BaseModel, ModelConfig
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel(BaseModel):
    """Base class for embedding models."""

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        raise NotImplementedError


class E5EmbeddingModel(EmbeddingModel):
    """E5 multilingual embedding model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.model_name = config.config.get("name", "intfloat/multilingual-e5-large-instruct")
        self.cache_dir = config.config.get("cache_dir", "./models/embeddings/e5_cache")
        self.device = config.config.get("device", "cpu")

    def load(self) -> None:
        """Load E5 model from HuggingFace."""
        if self._loaded:
            logger.info(f"E5 model already loaded: {self.model_name}")
            return

        logger.info(f"Loading E5 model: {self.model_name}")

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )

            self._loaded = True
            logger.info(f"E5 model loaded successfully (device: {self.device})")

        except Exception as e:
            logger.error(f"Failed to load E5 model: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self.model is not None

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate E5 embeddings."""
        if not self.is_loaded():
            self.load()

        # E5 requires "query: " prefix for queries
        if isinstance(texts, str):
            texts = [f"query: {texts}"]
        else:
            texts = [f"query: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model (placeholder for future)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = config.config.get("api_key")
        self.model_name = config.config.get("model", "text-embedding-ada-002")

    def load(self) -> None:
        """OpenAI models don't need loading."""
        self._loaded = True
        logger.info(f"OpenAI embedding model ready: {self.model_name}")

    def is_loaded(self) -> bool:
        """Always ready if API key is set."""
        return self._loaded

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate OpenAI embeddings (not implemented)."""
        raise NotImplementedError("OpenAI embeddings not yet implemented")

