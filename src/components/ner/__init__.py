"""
NER Component - Standalone Named Entity Recognition

Extracts entities from text using hybrid approach:
- Transformer-based NER (XLM-RoBERTa)
- Regex patterns for structured data
- Entity-to-slot mapping

Can be run independently or composed in a pipeline.
"""

from src.components.ner.component import (
    NERComponent,
    NERInput,
    NEROutput,
    NERConfig,
)

__all__ = [
    "NERComponent",
    "NERInput",
    "NEROutput",
    "NERConfig",
]
