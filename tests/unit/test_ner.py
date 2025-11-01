"""Tests for the NER extraction utilities in src."""

import pytest

from src.application.ner.ner_extractor import NERExtractor
from src.application.ner.entity_patterns import EntityPatterns


class DummyPipeline:
    """Simple stand-in for HuggingFace NER pipeline."""

    def __call__(self, text):
        return [
            {
                "entity_group": "PERSON",
                "word": "Rahim",
                "score": 0.92,
            },
            {
                "entity_group": "LOCATION",
                "word": "Dhaka",
                "score": 0.88,
            },
        ]


def _make_extractor(with_pipeline=False):
    extractor = NERExtractor(enable_regex_fallback=True)
    extractor._loaded = True  # Skip heavy model loading
    extractor.ner_pipeline = DummyPipeline() if with_pipeline else None
    return extractor


@pytest.mark.asyncio
async def test_extract_nid_17_digits():
    extractor = _make_extractor()
    result = await extractor.extract("আমার NID 12345678901234567 নম্বর।")

    assert result["entities"]["nid_number"] == "12345678901234567"
    assert result["method"] == "regex"


@pytest.mark.asyncio
async def test_extract_phone_bd_format():
    extractor = _make_extractor()
    result = await extractor.extract("ফোন করুন 01712345678 নম্বরে।")

    assert result["entities"]["phone"] == "01712345678"


@pytest.mark.asyncio
async def test_extract_email():
    extractor = _make_extractor()
    result = await extractor.extract("ইমেল পাঠান helpdesk@example.gov.bd ঠিকানায়।")

    assert result["entities"]["email"] == "helpdesk@example.gov.bd"


@pytest.mark.asyncio
async def test_extract_date_formats():
    extractor = _make_extractor()
    result = await extractor.extract("আমার জন্ম তারিখ 15-01-1990।")

    assert "date" in result["entities"]
    assert result["entities"]["date"] == "15-01-1990"


@pytest.mark.asyncio
async def test_multiple_entities_merges_regex_and_bert():
    extractor = _make_extractor(with_pipeline=True)
    text = "Rahim এর NID 12345678901234567 এবং তিনি Dhaka তে থাকেন।"
    result = await extractor.extract(text, form_slots=["full_name", "city", "nid_number"])

    assert result["entities"]["nid_number"] == "12345678901234567"
    assert result["entities"]["full_name"] == "Rahim"
    assert result["entities"]["city"] == "Dhaka"
    assert result["method"] == "hybrid"


@pytest.mark.asyncio
async def test_bert_fallback_skips_low_confidence(monkeypatch):
    class LowConfidencePipeline:
        def __call__(self, text):
            return [
                {"entity_group": "PERSON", "word": "Nope", "score": 0.2},
            ]

    extractor = _make_extractor()
    extractor.ner_pipeline = LowConfidencePipeline()

    result = await extractor.extract("এখানে কোন তথ্য নেই।")

    # Should ignore low confidence entity and fall back to regex-only method
    assert result["entities"] == {}
    assert result["method"] == "none"


@pytest.mark.asyncio
async def test_form_aware_extraction_maps_slots():
    extractor = _make_extractor()
    text = "আমার NID 12345678901234567 এবং ফোন 01712345678।"

    result = await extractor.extract(text, form_slots=["nid_number", "phone"])

    assert result["entities"]["nid_number"] == "12345678901234567"
    assert result["entities"]["phone"] == "01712345678"


@pytest.mark.asyncio
async def test_extract_for_slot_prefers_regex():
    extractor = _make_extractor()
    value = await extractor.extract_for_slot(
        "আমার একাউন্ট 912345678901", slot_name="account_number"
    )

    assert value == "912345678901"


def test_entity_patterns_prioritize_first_match():
    patterns = EntityPatterns()
    entities = patterns.extract_all(
        "NID: 12345678901234567 এবং 98765432109876543"
    )
    prioritized = patterns.prioritize_entities(entities)

    assert prioritized["nid_number"] == "12345678901234567"


def test_entity_patterns_validate():
    patterns = EntityPatterns()
    assert patterns.validate_entity("12345678901234567", "nid_number")
    assert not patterns.validate_entity("abc", "nid_number")


@pytest.mark.asyncio
async def test_extract_handles_no_entities():
    extractor = _make_extractor()
    result = await extractor.extract("শুধু সাধারণ টেক্সট।")

    assert result["entities"] == {}
    assert result["method"] == "none"
