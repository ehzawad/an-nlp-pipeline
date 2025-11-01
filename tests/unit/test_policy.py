"""Tests for policy handlers and PolicyEngine in src."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.application.policy.base_policy import Action, ActionType
from src.application.policy.handlers import (
    ActiveFormHandler,
    ClarificationFallbackHandler,
    EscalationHandler,
    FormTriggerHandler,
    HighConfidenceFAQHandler,
)
from src.application.policy.policy_engine import PolicyEngine
from src.application.policy.faq_policy import FAQPolicy
from src.application.policy.clarification_policy import ClarificationPolicy
from src.application.session.models import FormStateData, SessionState


def _make_session(session_id: str = "session") -> SessionState:
    """Helper to create a blank session state."""
    return SessionState(session_id=session_id)


# ---------------------------------------------------------------------------
# Handler unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_form_handler_continues():
    action = Action(
        action_type=ActionType.COLLECT_SLOT,
        data={"form_name": "dummy_form"},
        response_text="next prompt",
    )
    mock_policy = SimpleNamespace(decide=AsyncMock(return_value=action))
    handler = ActiveFormHandler(mock_policy)
    session_state = SimpleNamespace(active_form=FormStateData(form_name="dummy_form"))

    result = await handler.handle(session_state, "hi", {}, None)

    mock_policy.decide.assert_awaited_once()
    assert result is action


@pytest.mark.asyncio
async def test_form_trigger_handler_starts_form(empty_session_state, example_form_registry):
    handler = FormTriggerHandler(example_form_registry)
    nlp_result = {
        "results": [{"tag": "hotel_booking"}],
        "confidence": 0.9,
    }

    action = await handler.handle(empty_session_state, "Need a room", nlp_result, None)

    assert action is not None
    assert action.action_type == ActionType.START_FORM
    assert action.data["form_name"] == "hotel_booking"
    assert empty_session_state.active_form is not None
    assert empty_session_state.active_form.current_slot == "city"


@pytest.mark.asyncio
async def test_high_confidence_faq_handler():
    faq_policy = FAQPolicy(confidence_threshold=0.7)
    handler = HighConfidenceFAQHandler(faq_policy)
    session_state = SimpleNamespace(active_form=None)
    nlp_result = {
        "confidence": 0.95,
        "results": [{"tag": "nid_status", "answer": "Your status"}],
    }

    action = await handler.handle(session_state, "status?", nlp_result, None)

    assert action.action_type == ActionType.FAQ_ANSWER
    assert action.data["results"][0]["answer"] == "Your status"


@pytest.mark.asyncio
async def test_medium_confidence_clarification():
    handler = ClarificationFallbackHandler(ClarificationPolicy(min_confidence=0.5))
    session_state = SimpleNamespace(active_form=None)
    nlp_result = {
        "confidence": 0.6,
        "classification": [
            {"cluster": "status_check", "confidence": 0.6},
            {"cluster": "faq", "confidence": 0.3},
        ],
    }

    action = await handler.handle(session_state, "help", nlp_result, None)

    assert action.action_type == ActionType.CLARIFY_INTENT
    assert len(action.data["top_options"]) == 2


@pytest.mark.asyncio
async def test_low_confidence_escalation():
    handler = ClarificationFallbackHandler(ClarificationPolicy(min_confidence=0.5))
    session_state = SimpleNamespace(active_form=None)
    nlp_result = {"confidence": 0.2, "classification": []}

    action = await handler.handle(session_state, "??", nlp_result, None)

    assert action.action_type == ActionType.ESCALATE
    assert "Low confidence" in action.data["reason"]


@pytest.mark.asyncio
async def test_escalation_handler_no_flag_returns_none():
    handler = EscalationHandler()
    session_state = SimpleNamespace(active_form=None, escalation_flag=False)

    action = await handler.handle(session_state, "hi", {}, None)

    assert action is None


def test_chain_execution_order(example_form_registry):
    engine = PolicyEngine(enable_forms=True, form_registry=example_form_registry)
    chain = engine._handler_chain  # Escalation head of chain

    order = []
    current = chain
    while current:
        order.append(type(current).__name__)
        current = current._next_handler

    assert order == [
        "EscalationHandler",
        "ActiveFormHandler",
        "FormTriggerHandler",
        "HighConfidenceFAQHandler",
        "ClarificationFallbackHandler",
    ]


# ---------------------------------------------------------------------------
# PolicyEngine end-to-end decisions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_next_action_faq():
    engine = PolicyEngine(enable_forms=False)
    session_state = _make_session()
    nlp_result = {
        "confidence": 0.85,
        "results": [{"tag": "faq", "answer": "Answer"}],
    }

    action = await engine.decide_next_action(session_state, "question", nlp_result, None)

    assert action.action_type == ActionType.FAQ_ANSWER


@pytest.mark.asyncio
async def test_decide_next_action_form(example_form_registry):
    engine = PolicyEngine(enable_forms=True, form_registry=example_form_registry)
    session_state = _make_session()
    nlp_result = {
        "results": [{"tag": "hotel_booking"}],
        "confidence": 0.9,
    }

    action = await engine.decide_next_action(session_state, "book a hotel", nlp_result, None)

    assert action.action_type == ActionType.START_FORM
    assert session_state.active_form is not None
    assert session_state.active_form.form_name == "hotel_booking"


@pytest.mark.asyncio
async def test_decide_next_action_form_respects_active_form(example_form_registry):
    engine = PolicyEngine(enable_forms=True, form_registry=example_form_registry)
    # Ensure chain rebuilt with the provided registry
    session_state = _make_session()
    session_state.active_form = FormStateData(
        form_name="hotel_booking",
        current_slot="city",
        filled_slots={},
    )

    # Stub form policy to avoid executing real form logic
    engine.form_policy.decide = AsyncMock(
        return_value=Action(
            action_type=ActionType.COLLECT_SLOT,
            data={"form_name": "hotel_booking", "slot_name": "city"},
            response_text="Which city?",
        )
    )

    nlp_result = {
        "confidence": 0.95,
        "results": [{"tag": "hotel_booking"}],
    }

    action = await engine.decide_next_action(session_state, "Dhaka", nlp_result, None)

    engine.form_policy.decide.assert_awaited_once()
    assert action.action_type == ActionType.COLLECT_SLOT
    assert action.data["slot_name"] == "city"


@pytest.mark.asyncio
async def test_decide_next_action_clarify():
    engine = PolicyEngine(enable_forms=False)
    session_state = _make_session()
    nlp_result = {
        "confidence": 0.6,
        "classification": [{"cluster": "nid_status_check", "confidence": 0.6}],
        "results": [],
    }

    action = await engine.decide_next_action(session_state, "status?", nlp_result, None)

    assert action.action_type == ActionType.CLARIFY_INTENT


@pytest.mark.asyncio
async def test_decide_next_action_escalate():
    engine = PolicyEngine(enable_forms=False)
    session_state = _make_session()
    nlp_result = {"confidence": 0.1, "classification": [], "results": []}

    action = await engine.decide_next_action(session_state, "???", nlp_result, None)

    assert action.action_type == ActionType.ESCALATE


@pytest.mark.asyncio
async def test_decide_next_action_fallback_on_missing_nlp():
    engine = PolicyEngine(enable_forms=False)
    session_state = _make_session()

    action = await engine.decide_next_action(session_state, "hello", None, None)

    assert action.action_type == ActionType.FALLBACK
