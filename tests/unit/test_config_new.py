"""Tests for the ConfigLoader utility."""

import json
from pathlib import Path

import pytest

from src.application.config.loader import ConfigLoader


def _write_config(tmp_path: Path, name: str, data: dict) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    with open(config_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_config_loader_reads_file(tmp_path):
    _write_config(tmp_path, "main", {"a": 1})
    loader = ConfigLoader(config_dir=tmp_path / "config")

    config = loader.load("main")

    assert config["a"] == 1


def test_config_loader_applies_overrides(tmp_path):
    _write_config(tmp_path, "main", {"a": {"b": 1, "c": 2}})
    loader = ConfigLoader(config_dir=tmp_path / "config")

    config = loader.load("main", overrides={"a": {"c": 5}})

    assert config["a"]["b"] == 1
    assert config["a"]["c"] == 5


def test_config_loader_cache(tmp_path):
    _write_config(tmp_path, "main", {"x": 1})
    loader = ConfigLoader(config_dir=tmp_path / "config")

    first = loader.load("main")
    # Modify file directly; cached version should persist until cache cleared
    _write_config(tmp_path, "main", {"x": 2})
    second = loader.load("main")

    assert first == second == {"x": 1}

    loader.clear_cache()
    refreshed = loader.load("main")
    assert refreshed == {"x": 2}


def test_config_loader_resolve_path(tmp_path):
    loader = ConfigLoader(config_dir=tmp_path / "config")
    resolved = loader.resolve_path("models")

    assert resolved.is_absolute()
    assert str(resolved).endswith("models")

