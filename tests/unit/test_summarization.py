"""Tests for summarization utilities."""

import pytest

from src.application.summarization.passthrough_summarizer import PassthroughSummarizer


@pytest.mark.asyncio
async def test_passthrough_summarizer_returns_same_text():
    summarizer = PassthroughSummarizer()

    text = "Sample summary input."
    summarized, metadata = await summarizer.summarize(text)

    assert summarized == text
    assert metadata["summarized"] is False
    assert metadata["method"] == "passthrough"


@pytest.mark.asyncio
async def test_passthrough_summarizer_metadata_reason():
    summarizer = PassthroughSummarizer()
    _, metadata = await summarizer.summarize("Another input")

    assert metadata["reason"] == "passthrough_implementation"

