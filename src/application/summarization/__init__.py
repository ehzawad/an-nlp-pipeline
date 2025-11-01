"""Summarization module."""

from .base_summarizer import BaseSummarizer
from .passthrough_summarizer import PassthroughSummarizer

__all__ = [
    "BaseSummarizer",
    "PassthroughSummarizer",
]

