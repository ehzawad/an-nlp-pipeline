"""Tests for context augmentation components in src."""

import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import typing


def _load_fraction_classifier():
    module_name = "src.application.context.fraction_classifier"
    if module_name in sys.modules:
        return sys.modules[module_name]

    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "src" / "application" / "context" / "fraction_classifier.py"
    source = module_path.read_text(encoding="utf-8")

    module = types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.Any = typing.Any  # Predefine Any before executing module
    sys.modules[module_name] = module
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


fraction_module = _load_fraction_classifier()
FractionalQueryClassifier = fraction_module.FractionalQueryClassifier

from src.application.context.augmenter import ContextAugmenter
from src.application.session.models import SessionState


class PickleableClassifier:
    """Lightweight classifier compatible with FractionalQueryClassifier expectations."""

    def __init__(self, fractional_probability: float):
        self.fractional_probability = fractional_probability

    def predict_proba(self, features):
        return np.array([[1 - self.fractional_probability, self.fractional_probability]])

    def predict(self, features):
        return np.array([1 if self.fractional_probability >= 0.5 else 0])


class StubEmbeddingModel:
    """Deterministic embedding model for tests."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        # Encode sentence length to float vector
        embeddings = []
        for text in texts:
            embeddings.append([float(len(text))])
        return np.array(embeddings, dtype=float)


def _write_classifier(tmp_path: Path, probability: float) -> Path:
    model_path = tmp_path / f"classifier_{probability:.2f}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(PickleableClassifier(probability), f)
    return model_path


def _write_mappings(tmp_path: Path, mappings: dict) -> Path:
    mapping_path = tmp_path / "context_mappings.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f)
    return mapping_path


@pytest.mark.asyncio
async def test_fractional_classifier_detects_incomplete(tmp_path):
    model_path = _write_classifier(tmp_path, probability=0.85)
    classifier = FractionalQueryClassifier(model_path=model_path, embedding_model=StubEmbeddingModel())

    assert await classifier.is_fractional("status", threshold=0.7) is True
    assert await classifier.predict("status") == 1


@pytest.mark.asyncio
async def test_fractional_classifier_detects_complete(tmp_path):
    model_path = _write_classifier(tmp_path, probability=0.2)
    classifier = FractionalQueryClassifier(model_path=model_path, embedding_model=StubEmbeddingModel())

    assert await classifier.is_fractional("complete question", threshold=0.7) is False
    assert await classifier.predict("complete question") == 0


@pytest.mark.asyncio
async def test_augmenter_expands_fractional_query(tmp_path):
    model_path = _write_classifier(tmp_path, probability=0.9)
    classifier = FractionalQueryClassifier(model_path=model_path, embedding_model=StubEmbeddingModel())
    mapping_path = _write_mappings(tmp_path, {"nid_status": "জাতীয় পরিচয়পত্রের অবস্থা"})

    augmenter = ContextAugmenter(
        classifier=classifier,
        mappings_file=str(mapping_path),
        context_window_turns=1,
        min_confidence=0.7,
    )

    session = SessionState(session_id="s1")
    session.last_context_tag = "nid_status"

    augmented, metadata = await augmenter.augment_if_needed("কোথায়", session)

    assert augmented.startswith("জাতীয় পরিচয়পত্রের অবস্থা")
    assert metadata["augmented"] is True
    assert metadata["context_tag"] == "nid_status"


@pytest.mark.asyncio
async def test_augmenter_skips_complete_query(tmp_path):
    model_path = _write_classifier(tmp_path, probability=0.2)
    classifier = FractionalQueryClassifier(model_path=model_path, embedding_model=StubEmbeddingModel())
    mapping_path = _write_mappings(tmp_path, {"nid_status": "জাতীয় পরিচয়পত্রের অবস্থা"})

    augmenter = ContextAugmenter(
        classifier=classifier,
        mappings_file=str(mapping_path),
        min_confidence=0.7,
    )

    session = SessionState(session_id="s1")
    session.last_context_tag = "nid_status"

    augmented, metadata = await augmenter.augment_if_needed("সম্পূর্ণ প্রশ্ন", session)

    assert augmented == "সম্পূর্ণ প্রশ্ন"
    assert metadata == {}


@pytest.mark.asyncio
async def test_augmenter_reports_missing_mapping(tmp_path):
    model_path = _write_classifier(tmp_path, probability=0.8)
    classifier = FractionalQueryClassifier(model_path=model_path, embedding_model=StubEmbeddingModel())
    mapping_path = _write_mappings(tmp_path, {})  # No mappings available

    augmenter = ContextAugmenter(
        classifier=classifier,
        mappings_file=str(mapping_path),
    )

    session = SessionState(session_id="s1")
    session.last_context_tag = "unknown_tag"

    augmented, metadata = await augmenter.augment_if_needed("কি হচ্ছে", session)

    assert augmented == "কি হচ্ছে"
    assert metadata["no_mapping"] is True


@pytest.mark.asyncio
async def test_augmenter_uses_history_when_last_tag_missing(tmp_path):
    model_path = _write_classifier(tmp_path, probability=0.9)
    classifier = FractionalQueryClassifier(model_path=model_path, embedding_model=StubEmbeddingModel())
    mapping_path = _write_mappings(tmp_path, {"status_context": "পূর্ববর্তী অবস্থা"})

    augmenter = ContextAugmenter(
        classifier=classifier,
        mappings_file=str(mapping_path),
        context_window_turns=2,
    )

    class SimpleSession:
        conversation_history = [
            {"context_tag": "random"},
            {"context_tag": "status_context"},
        ]

    augmented, metadata = await augmenter.augment_if_needed("এখন?", SimpleSession())

    assert metadata.get("context_tag") == "status_context", metadata
    assert metadata.get("augmented") is True, metadata
    assert augmented.startswith("পূর্ববর্তী অবস্থা"), metadata
