"""
Semantic Search Component - Standalone FAISS-based Search

Searches for similar questions using FAISS indices and E5 embeddings.

Can be run independently or composed in a pipeline.
"""

from src.components.semantic_search.component import (
    SemanticSearchComponent,
    SemanticSearchInput,
    SemanticSearchOutput,
    SemanticSearchConfig,
)

__all__ = [
    "SemanticSearchComponent",
    "SemanticSearchInput",
    "SemanticSearchOutput",
    "SemanticSearchConfig",
]
