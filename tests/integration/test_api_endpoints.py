"""Integration tests for FastAPI endpoints (NO AUTHENTICATION)."""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from src.interfaces.api.app import app, dialogue_pipelines


@contextmanager
def client_with_overrides(overrides: dict):
    original = app.dependency_overrides.copy()
    with TestClient(app) as client:
        app.dependency_overrides.update(overrides)
        try:
            yield client
        finally:
            app.dependency_overrides = original


def test_health_endpoint():
    """Test health check endpoint"""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"


def test_root_endpoint():
    """Test root endpoint returns service info"""
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "NID Support Dialogue System"
        assert body["version"] == "2.0.0"
        assert "docs" in body


@pytest.mark.parametrize("pipeline_present", [True, False])
def test_chat_endpoint_handles_pipeline(pipeline_present):
    """Test chat endpoint with and without pipeline"""
    class StubPipeline:
        def __init__(self):
            self.calls = []

        async def process_turn(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "text": "Hello from stub",
                "metadata": {"foo": "bar"},
                "success": True,
            }

    pipeline = StubPipeline()

    with TestClient(app) as client:
        if pipeline_present:
            dialogue_pipelines["default"] = pipeline
        else:
            dialogue_pipelines.pop("default", None)

        resp = client.post(
            "/api/v2/chat",
            json={"query": "Hello"},
        )

        body = resp.json()
        assert resp.status_code == 200

        if pipeline_present:
            assert body["success"] is True
            assert body["text"] == "Hello from stub"
            assert body["metadata"]["foo"] == "bar"
            assert "session_id" in body
            # Verify pipeline was called
            assert len(pipeline.calls) == 1
            assert pipeline.calls[0]["query"] == "Hello"
        else:
            assert body["success"] is False
            assert "error" in body["metadata"]
            assert body["metadata"]["error"] == "pipeline_not_found"

        dialogue_pipelines.pop("default", None)


def test_chat_endpoint_accepts_session_id():
    """Test that chat endpoint accepts optional session_id"""
    class StubPipeline:
        async def process_turn(self, **kwargs):
            return {
                "text": "Response",
                "metadata": {},
                "success": True,
            }

    with TestClient(app) as client:
        dialogue_pipelines["default"] = StubPipeline()

        # Test without session_id (should be auto-generated)
        resp1 = client.post("/api/v2/chat", json={"query": "Hello"})
        body1 = resp1.json()
        assert "session_id" in body1
        session_id_1 = body1["session_id"]

        # Test with explicit session_id
        resp2 = client.post(
            "/api/v2/chat",
            json={"query": "Hello", "session_id": "my-custom-session"},
        )
        body2 = resp2.json()
        assert body2["session_id"] == "my-custom-session"

        # Test that different calls get different auto-generated session_ids
        resp3 = client.post("/api/v2/chat", json={"query": "Hello"})
        body3 = resp3.json()
        session_id_3 = body3["session_id"]
        assert session_id_3 != session_id_1

        dialogue_pipelines.pop("default", None)
