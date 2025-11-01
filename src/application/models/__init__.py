"""Model abstraction layer."""

from .base import BaseModel, ModelConfig
from .embeddings import EmbeddingModel, E5EmbeddingModel, OpenAIEmbeddingModel
from .classifiers import ClassifierModel
from .search import SemanticSearchModel
from .registry import ModelRegistry

__all__ = [
    "BaseModel",
    "ModelConfig",
    "EmbeddingModel",
    "E5EmbeddingModel",
    "OpenAIEmbeddingModel",
    "ClassifierModel",
    "SemanticSearchModel",
    "ModelRegistry",
]

