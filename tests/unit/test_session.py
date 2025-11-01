"""Tests for session state, store, and manager components."""

from datetime import datetime, timedelta

import pytest

from src.application.session.models import SessionState, FormStateData
from src.application.session.store import InMemorySessionStore
from src.application.session.manager import SessionManager


@pytest.fixture
def session_store_fake():
    """Create in-memory session store for testing."""
    return InMemorySessionStore(ttl_seconds=10)


def test_session_state_add_turn_trims_history():
    session = SessionState(session_id="s1", max_history=2)

    session.add_turn("q1", "a1")
    session.add_turn("q2", "a2")
    session.add_turn("q3", "a3")

    assert len(session.conversation_history) == 2
    assert session.conversation_history[0].user_query == "q2"


def test_session_state_increment_error_sets_escalation():
    session = SessionState(session_id="s1", max_errors=2)

    session.increment_error()
    assert session.escalation_flag is False

    session.increment_error()
    assert session.escalation_flag is True


def test_session_state_serialization_roundtrip():
    session = SessionState(session_id="s1")
    session.user_id = "user-1"
    session.last_context_tag = "nid_status"
    session.active_form = FormStateData(form_name="nid_status_check", current_slot="nid_number")
    session.add_turn("hello", "hi there")

    serialized = session.to_dict()
    restored = SessionState.from_dict(serialized)

    assert restored.session_id == "s1"
    assert restored.active_form.form_name == "nid_status_check"
    assert restored.conversation_history[0].user_query == "hello"


def test_session_state_in_form_state():
    session = SessionState(session_id="s1")
    assert session.in_form_state() is False

    session.active_form = FormStateData(form_name="form")
    assert session.in_form_state() is True


@pytest.mark.asyncio
async def test_session_store_set_get_delete(session_store_fake):
    session = SessionState(session_id="alpha", user_id="user-1")

    success = await session_store_fake.set("alpha", session)
    assert success is True

    loaded = await session_store_fake.get("alpha")
    assert loaded is not None
    assert loaded.session_id == "alpha"

    exists = await session_store_fake.exists("alpha")
    assert exists is True

    deleted = await session_store_fake.delete("alpha")
    assert deleted is True

    missing = await session_store_fake.get("alpha")
    assert missing is None


@pytest.mark.asyncio
async def test_session_store_touch(session_store_fake):
    session = SessionState(session_id="beta")
    await session_store_fake.set("beta", session)

    assert await session_store_fake.touch("beta") is True


@pytest.mark.asyncio
async def test_session_manager_get_or_create(session_store_fake):
    manager = SessionManager(session_store_fake)

    created = await manager.get_or_create("s1", user_id="user-xyz")
    assert created.session_id == "s1"
    assert created.user_id == "user-xyz"

    # Second call should fetch existing session and keep user id
    fetched = await manager.get_or_create("s1")
    assert fetched.user_id == "user-xyz"


@pytest.mark.asyncio
async def test_session_manager_update_persists_changes(session_store_fake):
    manager = SessionManager(session_store_fake)
    session = await manager.get_or_create("s2")

    session.last_context_tag = "nid_status"
    success = await manager.update(session)
    assert success is True

    loaded = await session_store_fake.get("s2")
    assert loaded.last_context_tag == "nid_status"


@pytest.mark.asyncio
async def test_session_manager_delete(session_store_fake):
    manager = SessionManager(session_store_fake)
    session = await manager.get_or_create("s3")
    await manager.update(session)

    assert await manager.delete("s3") is True
    assert await session_store_fake.get("s3") is None


def test_session_state_touch_updates_timestamps():
    session = SessionState(session_id="time-test")
    before = session.last_activity
    session.add_turn("?", "!")
    assert session.last_activity >= before


@pytest.mark.asyncio
async def test_session_store_ttl_expiration(session_store_fake):
    """Test that sessions expire after TTL."""
    session = SessionState(session_id="expire_test")
    await session_store_fake.set("expire_test", session)
    
    # Session should exist
    assert await session_store_fake.exists("expire_test") is True
    
    # Wait for expiration (TTL is 10 seconds in fixture)
    import asyncio
    await asyncio.sleep(11)
    
    # Session should be expired
    assert await session_store_fake.get("expire_test") is None
    assert await session_store_fake.exists("expire_test") is False


@pytest.mark.asyncio
async def test_session_store_count(session_store_fake):
    """Test count method."""
    assert session_store_fake.count() == 0
    
    await session_store_fake.set("c1", SessionState(session_id="c1"))
    await session_store_fake.set("c2", SessionState(session_id="c2"))
    
    assert session_store_fake.count() == 2
    
    await session_store_fake.delete("c1")
    assert session_store_fake.count() == 1
