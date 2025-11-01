"""Shared fixtures for tests targeting src."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pytest

# Ensure repository root is on sys.path so src can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.forms.form_registry import FormRegistry
from src.application.forms.form_runner import FormRunner
from src.application.forms.examples.bank_account_verification import (
    BankAccountVerificationForm,
)
from src.application.forms.examples.correction import CorrectionRequestForm
from src.application.forms.examples.hotel_booking import HotelBookingForm
from src.application.forms.examples.nid_status import NIDStatusCheckForm
from src.application.session.models import FormStateData, SessionState
from src.application.hooks.manager import HookManager, HookPoint


@pytest.fixture
def example_form_registry() -> FormRegistry:
    """Registry preloaded with the example forms from src."""
    registry = FormRegistry()
    registry.register(NIDStatusCheckForm())
    registry.register(CorrectionRequestForm())
    registry.register(HotelBookingForm())
    registry.register(BankAccountVerificationForm())
    return registry


@pytest.fixture
def form_runner() -> FormRunner:
    """Provide a form runner with default retry settings."""
    return FormRunner()


@pytest.fixture
def empty_session_state() -> SessionState:
    """Create a blank session state for tests."""
    return SessionState(session_id="test-session")


@pytest.fixture
def active_form_state() -> SessionState:
    """Session state with an active dummy form for policy tests."""
    session = SessionState(session_id="session-with-form")
    session.active_form = FormStateData(
        form_name="nid_status_check",
        current_slot="nid_number",
        filled_slots={},
        slot_retries={},
        interruption_count=0,
    )
    return session


@pytest.fixture
def sample_nlp_result() -> Dict[str, Dict]:
    """Minimal NLP result with a single high-confidence hit."""
    return {
        "confidence": 0.92,
        "results": [
            {
                "tag": "correction_application_status",
                "answer": "Sample answer",
                "confidence": 0.92,
            }
        ],
        "classification": [
            {
                "cluster": "nid_status_check",
                "confidence": 0.92,
            }
        ],
    }


@pytest.fixture
def timestamp() -> datetime:
    """Deterministic timestamp for tests that need ordering."""
    return datetime(2025, 1, 1, 12, 0, 0)


@pytest.fixture
def hook_manager_new() -> HookManager:
    """Provide a fresh HookManager for src hooks tests."""
    manager = HookManager()
    yield manager
    manager.clear()


@pytest.fixture
def sample_hook_context():
    """Minimal hook context dictionary used across hook tests."""
    return {
        "query": "hello",
        "session_id": "session-1",
        "metadata": {},
    }

