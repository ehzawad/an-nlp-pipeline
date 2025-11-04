"""
Classification Component - Standalone Intent Classification

Classifies user queries into predefined intent clusters.
Uses E5 embeddings + Logistic Regression.

Can be run independently or composed in a pipeline.
"""

from src.components.classification.component import (
    ClassificationComponent,
    ClassificationInput,
    ClassificationOutput,
    ClassificationConfig,
)

__all__ = [
    "ClassificationComponent",
    "ClassificationInput",
    "ClassificationOutput",
    "ClassificationConfig",
]
