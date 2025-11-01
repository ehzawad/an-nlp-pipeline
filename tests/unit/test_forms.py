"""Tests for the form framework in src."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from src.application.forms.base_form import BaseForm, BaseSlot
from src.application.forms.form_registry import FormRegistry
from src.application.forms.form_runner import FormRunner, FormStatus
from src.application.forms.examples.bank_account_verification import (
    BankAccountVerificationForm,
)
from src.application.forms.examples.correction import CorrectionRequestForm
from src.application.forms.examples.hotel_booking import HotelBookingForm
from src.application.forms.examples.nid_status import NIDStatusCheckForm


FUTURE_DATE = (datetime.now(timezone.utc) + timedelta(days=365)).strftime("%d-%m-%Y")


class DummyForm(BaseForm):
    """Simple form for exercising BaseForm and FormRunner behaviour."""

    def __init__(self):
        self._execute_calls: List[Dict[str, Any]] = []
        self._slots = [
            BaseSlot(
                name="first_slot",
                prompt="enter first",
                validation_regex=r"^[A-Za-z]+$",
                error_message="letters only",
                max_retries=2,
            ),
            BaseSlot(
                name="second_slot",
                prompt="enter second",
                slot_type="numeric",
                error_message="numbers only",
                max_retries=2,
            ),
        ]

    @property
    def form_name(self) -> str:
        return "dummy_form"

    @property
    def trigger_tags(self) -> List[str]:
        return ["dummy_trigger"]

    def required_slots(self) -> List[BaseSlot]:
        return self._slots

    async def execute(self, filled_slots: Dict[str, Any]) -> Dict[str, Any]:
        self._execute_calls.append(filled_slots.copy())
        return {"success": True, "message": "executed", "data": filled_slots.copy()}


# ---------------------------------------------------------------------------
# BaseSlot validation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_regex_passes():
    slot = BaseSlot(
        name="nid_number",
        prompt="enter nid",
        validation_regex=r"^\d{4}$",
        error_message="invalid",
    )

    is_valid, error = await slot.validate_input("1234")

    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_validation_regex_fails():
    slot = BaseSlot(
        name="nid_number",
        prompt="enter nid",
        validation_regex=r"^\d{4}$",
        error_message="invalid",
    )

    is_valid, error = await slot.validate_input("abc")

    assert is_valid is False
    assert error == "invalid"


@pytest.mark.asyncio
async def test_async_validation_executes():
    called = asyncio.Event()

    async def validator(value: str) -> bool:
        called.set()
        await asyncio.sleep(0)
        return value == "good"

    slot = BaseSlot(
        name="async_slot",
        prompt="enter value",
        validation_func=validator,
        error_message="bad",
    )

    is_valid, error = await slot.validate_input("bad")

    assert called.is_set()
    assert is_valid is False
    assert error == "bad"


@pytest.mark.asyncio
async def test_custom_validator_called():
    calls = {"count": 0}

    def validator(value: str) -> bool:
        calls["count"] += 1
        return value == "ok"

    slot = BaseSlot(
        name="custom_slot",
        prompt="enter value",
        validation_func=validator,
        error_message="bad input",
    )

    is_valid, _ = await slot.validate_input("ok")

    assert is_valid is True
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_slot_without_validation_accepts_all():
    slot = BaseSlot(
        name="free_slot",
        prompt="say anything",
        required=False,
    )

    is_valid, error = await slot.validate_input({"any": "thing"})

    assert is_valid is True
    assert error is None


# ---------------------------------------------------------------------------
# BaseForm and FormRunner interaction tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_form_activation():
    form = DummyForm()
    runner = FormRunner()

    state = await runner.start_form(form)

    assert state.form_name == "dummy_form"
    assert state.status == FormStatus.COLLECTING
    assert state.current_slot == "first_slot"


@pytest.mark.asyncio
async def test_collect_slot_valid_input():
    form = DummyForm()
    runner = FormRunner()
    state = await runner.start_form(form)

    # Provide valid value for first slot and move to next
    state, response = await runner.collect_slot(form, state, "Alpha")

    assert state.current_slot == "second_slot"
    assert state.filled_slots["first_slot"] == "Alpha"
    assert "enter second" in response


@pytest.mark.asyncio
async def test_collect_slot_invalid_input():
    form = DummyForm()
    runner = FormRunner()
    state = await runner.start_form(form)

    state, message = await runner.collect_slot(form, state, "123")

    assert state.status == FormStatus.COLLECTING
    assert state.slot_retries["first_slot"] == 1
    assert "letters only" in message


@pytest.mark.asyncio
async def test_validate_slot_success():
    form = DummyForm()

    is_valid, error = await form.validate_slot("first_slot", "Valid")

    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_form_completion():
    form = DummyForm()
    runner = FormRunner()
    state = await runner.start_form(form)

    state, _ = await runner.collect_slot(form, state, "Alpha")
    state, completion_message = await runner.collect_slot(form, state, "42")

    assert state.status == FormStatus.COMPLETE
    assert state.current_slot is None
    assert "প্রক্রিয়াকরণ" in completion_message or completion_message is None


@pytest.mark.asyncio
async def test_execute_calls_implementation():
    form = DummyForm()
    runner = FormRunner()
    state = await runner.start_form(form)
    state, _ = await runner.collect_slot(form, state, "Alpha")
    state, _ = await runner.collect_slot(form, state, "42")

    state, result = await runner.execute_form(form, state)

    assert state.status == FormStatus.EXECUTED
    assert result["success"] is True
    assert form._execute_calls and form._execute_calls[-1]["second_slot"] == "42"


@pytest.mark.asyncio
async def test_execute_form_requires_completion():
    form = DummyForm()
    runner = FormRunner()
    state = await runner.start_form(form)

    state, result = await runner.execute_form(form, state)

    assert result["success"] is False
    assert "ফর্ম সম্পন্ন নয়" in result["message"]


@pytest.mark.asyncio
async def test_form_runner_max_retries_failure():
    form = DummyForm()
    runner = FormRunner(max_retries=2)
    state = await runner.start_form(form)

    # Fail twice (form runner capped at 2 retries)
    for _ in range(2):
        state, message = await runner.collect_slot(form, state, "123")

    assert state.status == FormStatus.FAILED
    assert "সর্বাধিক প্রচেষ্টা" in message


@pytest.mark.asyncio
async def test_cancel_form_returns_message():
    form = DummyForm()
    runner = FormRunner()
    state = await runner.start_form(form)

    state, message = runner.cancel_form(form, state)

    assert state.status == FormStatus.CANCELLED
    assert message == "ফর্ম বাতিল করা হয়েছে।"


# ---------------------------------------------------------------------------
# Example form behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nid_status_form_validates_nid_length():
    form = NIDStatusCheckForm()

    is_valid, error = await form.validate_slot("nid_number", "123")
    assert is_valid is False
    assert "17 সংখ্যার" in error

    is_valid, error = await form.validate_slot("nid_number", "12345678901234567")
    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_nid_status_form_validates_date_format():
    form = NIDStatusCheckForm()

    is_valid, error = await form.validate_slot("date_of_birth", "1990-01-15")
    assert is_valid is False
    assert "DD-MM-YYYY" in error

    is_valid, error = await form.validate_slot("date_of_birth", "15-01-1990")
    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_nid_status_form_execute_returns_status():
    form = NIDStatusCheckForm()
    payload = {
        "nid_number": "12345678901234560",
        "date_of_birth": "15-01-1990",
    }

    result = await form.execute(payload)

    assert result["success"] is True
    assert result["nid_number"] == payload["nid_number"]
    assert "স্ট্যাটাস" in result["message"]


@pytest.mark.asyncio
async def test_correction_form_requires_valid_type():
    form = CorrectionRequestForm()

    is_valid, error = await form.validate_slot("correction_type", "invalid")
    assert is_valid is False
    assert "লিখুন" in error

    is_valid, error = await form.validate_slot("correction_type", "নাম")
    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_correction_form_requires_details_length():
    form = CorrectionRequestForm()

    is_valid, error = await form.validate_slot("correction_details", "short")
    assert is_valid is False
    assert "বিস্তারিত" in error

    is_valid, error = await form.validate_slot(
        "correction_details",
        "spelling mistake in last name",
    )
    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_correction_form_execute_returns_reference(monkeypatch):
    form = CorrectionRequestForm()

    # Stabilise timestamp for deterministic reference number
    monkeypatch.setattr("time.time", lambda: 1_725_897_600)

    payload = {
        "nid_number": "12345678901234567",
        "correction_type": "নাম",
        "correction_details": "spelling mistake in full name",
    }

    result = await form.execute(payload)

    assert result["success"] is True
    assert result["reference_number"].startswith("CORR-")
    assert "সংশোধন অনুরোধ" in result["message"]


@pytest.mark.asyncio
async def test_hotel_booking_form_validations():
    form = HotelBookingForm()

    is_valid, _ = await form.validate_slot("city", "Dhaka")
    assert is_valid is True

    is_valid, _ = await form.validate_slot("check_in_date", FUTURE_DATE)
    assert is_valid is True

    is_valid, _ = await form.validate_slot("num_guests", "3")
    assert is_valid is True

    is_valid, error = await form.validate_slot("num_guests", "20")
    assert is_valid is False
    assert "number between 1 and 10" in error


@pytest.mark.asyncio
async def test_hotel_booking_form_execute_builds_summary(monkeypatch):
    form = HotelBookingForm()

    # Freeze time for deterministic booking reference
    monkeypatch.setattr("time.time", lambda: 1_725_897_600)

    payload = {
        "city": "Dhaka",
        "check_in_date": FUTURE_DATE,
        "num_guests": "2",
        "full_name": "John Doe",
    }

    result = await form.execute(payload)

    assert result["success"] is True
    assert result["booking_reference"].startswith("HB-")
    assert "Booking confirmed" in result["message"]
    assert result["total_amount"] > 0


@pytest.mark.asyncio
async def test_bank_account_form_async_validation():
    form = BankAccountVerificationForm()

    is_valid, error = await form.validate_slot("account_number", "1234567890")
    assert is_valid is False
    assert "invalid" in error

    is_valid, error = await form.validate_slot("account_number", "912345678901")
    assert is_valid is True
    assert error is None


@pytest.mark.asyncio
async def test_bank_account_form_execute_returns_metadata(monkeypatch):
    form = BankAccountVerificationForm()

    payload = {
        "account_number": "912345678901",
        "phone": "01712345678",
    }

    result = await form.execute(payload)

    assert result["success"] is True
    assert result["verification_id"].startswith("VER-")
    assert "successfully verified" in result["message"]


# ---------------------------------------------------------------------------
# Registry and runner end-to-end tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_form_registry_register_and_get(example_form_registry):
    form = example_form_registry.get("nid_status_check")
    assert isinstance(form, NIDStatusCheckForm)


def test_form_registry_rejects_duplicate_names():
    registry = FormRegistry()
    registry.register(NIDStatusCheckForm())

    with pytest.raises(ValueError):
        registry.register(NIDStatusCheckForm())


def test_form_registry_rejects_tag_collisions():
    class DuplicateTagForm(BaseForm):
        @property
        def form_name(self) -> str:
            return "duplicate"

        @property
        def trigger_tags(self) -> List[str]:
            return ["correction_application_status"]

        def required_slots(self) -> List[BaseSlot]:
            return []

        async def execute(self, filled_slots: Dict[str, Any]) -> Dict[str, Any]:
            return {"success": True}

    registry = FormRegistry()
    registry.register(NIDStatusCheckForm())

    with pytest.raises(ValueError):
        registry.register(DuplicateTagForm())


@pytest.mark.asyncio
async def test_form_runner_handles_multiple_slots(example_form_registry):
    form = example_form_registry.get("hotel_booking")
    runner = FormRunner()

    state = await runner.start_form(form)
    state, _ = await runner.collect_slot(form, state, "Dhaka")
    state, _ = await runner.collect_slot(form, state, FUTURE_DATE)
    state, _ = await runner.collect_slot(form, state, "2")
    state, message = await runner.collect_slot(form, state, "John Doe")

    assert state.status == FormStatus.COMPLETE
    assert "সকল তথ্য" in (message or "")


@pytest.mark.asyncio
async def test_form_runner_executes_registered_form(example_form_registry):
    form = example_form_registry.get("nid_status_check")
    runner = FormRunner()

    state = await runner.start_form(form)
    state, _ = await runner.collect_slot(form, state, "12345678901234567")
    state, _ = await runner.collect_slot(form, state, "15-01-1990")
    state, result = await runner.execute_form(form, state)

    assert state.status == FormStatus.EXECUTED
    assert result["success"] is True
    assert "স্ট্যাটাস" in result["message"]
