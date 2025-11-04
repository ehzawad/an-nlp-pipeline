"""
LLM Component - Low-cost Abstractive Language Model

Provides text generation capabilities for:
- Summarization
- Query expansion
- Response generation
- Text rewriting

Supports multiple backends:
- Local models (Flan-T5, Phi-3, etc.)
- OpenAI API (with caching)
- Anthropic API (with caching)

Can be run independently or composed in a pipeline.
"""

from src.components.llm.component import (
    LLMComponent,
    LLMInput,
    LLMOutput,
    LLMConfig,
    LLMBackend,
)

__all__ = [
    "LLMComponent",
    "LLMInput",
    "LLMOutput",
    "LLMConfig",
    "LLMBackend",
]
