"""Tests for model base classes and registry."""

import pytest

from src.application.models.base import BaseModel, ModelConfig
from src.application.models.registry import ModelRegistry


class DummyModel(BaseModel):
    """Minimal BaseModel implementation for testing."""

    def __init__(self, name: str = "dummy"):
        super().__init__(ModelConfig(name=name, type="dummy", implementation="tests.Dummy"))
        self.load_count = 0

    def load(self) -> None:
        self.load_count += 1
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded


def test_base_model_reload_calls_load_once():
    model = DummyModel()
    model.reload()

    assert model.is_loaded() is True
    assert model.load_count == 1


def test_base_model_unload():
    model = DummyModel()
    model.load()
    assert model.is_loaded() is True

    model.unload()
    assert model.is_loaded() is False


def test_model_registry_register_and_get():
    registry = ModelRegistry()
    model = DummyModel("alpha")
    registry.register("alpha", model)

    fetched = registry.get("alpha")
    assert fetched is model
    assert fetched.is_loaded() is True
    assert fetched.load_count == 1


def test_model_registry_load_all():
    registry = ModelRegistry()
    models = [DummyModel(str(i)) for i in range(3)]
    for model in models:
        registry.register(model.config.name, model)

    registry.load_all()

    assert all(m.is_loaded() for m in models)
    assert all(m.load_count == 1 for m in models)


def test_model_registry_unload_all():
    registry = ModelRegistry()
    models = [DummyModel(str(i)) for i in range(2)]
    for model in models:
        registry.register(model.config.name, model)

    registry.load_all()
    registry.unload_all()

    assert all(m.is_loaded() is False for m in models)


def test_model_registry_list_models():
    registry = ModelRegistry()
    model = DummyModel("beta")
    registry.register("beta", model)

    status = registry.list_models()
    assert status["beta"] is False

    registry.get("beta")
    status = registry.list_models()
    assert status["beta"] is True


def test_model_registry_repr():
    registry = ModelRegistry()
    registry.register("a", DummyModel("a"))
    registry.register("b", DummyModel("b"))

    representation = repr(registry)
    assert "ModelRegistry" in representation

