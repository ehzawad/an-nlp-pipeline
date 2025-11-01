"""Form runner for slot collection and execution."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from .base_form import BaseForm, BaseSlot


class FormStatus(str, Enum):
    """Form execution status."""
    ACTIVE = "active"
    COLLECTING = "collecting"
    COMPLETE = "complete"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FormState:
    """State for an active form."""

    form_name: str
    status: FormStatus
    current_slot: Optional[str] = None
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    slot_retries: Dict[str, int] = field(default_factory=dict)
    execution_result: Optional[Dict] = None
    error_message: Optional[str] = None
    interruption_count: int = 0


class FormRunner:
    """
    Manages form slot collection and execution.

    Handles:
    - Sequential slot collection
    - Validation and retries
    - Form execution
    - State management
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize form runner.

        Args:
            max_retries: Maximum retry attempts per slot
        """
        self.max_retries = max_retries

    async def start_form(self, form: BaseForm) -> FormState:
        """
        Start a new form.

        Args:
            form: Form instance to start

        Returns:
            Initial FormState
        """
        state = FormState(
            form_name=form.form_name,
            status=FormStatus.ACTIVE
        )

        # Get first slot to collect
        slots = form.required_slots()
        if slots:
            state.current_slot = slots[0].name
            state.status = FormStatus.COLLECTING

        return state

    async def collect_slot(
        self,
        form: BaseForm,
        state: FormState,
        value: Any
    ) -> tuple[FormState, str]:
        """
        Collect value for current slot.

        Args:
            form: Form instance
            state: Current form state
            value: User-provided value

        Returns:
            tuple: (updated_state, response_message)
        """
        if not state.current_slot:
            return state, "ত্রুটি: কোন স্লট সক্রিয় নেই।"

        # Validate slot value
        is_valid, error_msg = await form.validate_slot(state.current_slot, value)

        if not is_valid:
            # Validation failed
            retry_count = state.slot_retries.get(state.current_slot, 0) + 1
            state.slot_retries[state.current_slot] = retry_count

            if retry_count >= self.max_retries:
                # Max retries exceeded
                state.status = FormStatus.FAILED
                state.error_message = f"সর্বাধিক প্রচেষ্টা অতিক্রম করেছে: {state.current_slot}"
                return state, state.error_message

            # Ask again
            slot = form.get_slot_by_name(state.current_slot)
            retry_msg = f"{error_msg or 'অবৈধ ইনপুট।'}\n{slot.prompt}"
            return state, retry_msg

        # Validation successful - store value
        state.filled_slots[state.current_slot] = value
        state.slot_retries[state.current_slot] = 0

        # Move to next slot
        next_slot = self._get_next_slot(form, state)

        if next_slot:
            # More slots to collect
            state.current_slot = next_slot.name
            return state, next_slot.prompt
        else:
            # All slots filled
            state.status = FormStatus.COMPLETE
            state.current_slot = None

            # Call on_form_complete hook
            completion_msg = await form.on_form_complete(state.filled_slots)
            return state, completion_msg or "সকল তথ্য সংগৃহীত হয়েছে। প্রক্রিয়াকরণ..."

    async def execute_form(
        self,
        form: BaseForm,
        state: FormState
    ) -> tuple[FormState, Dict[str, Any]]:
        """
        Execute form action with filled slots.

        Args:
            form: Form instance
            state: Form state (must be COMPLETE)

        Returns:
            tuple: (updated_state, execution_result)
        """
        if state.status != FormStatus.COMPLETE:
            return state, {
                "success": False,
                "message": f"ফর্ম সম্পন্ন নয়: {state.status}"
            }

        try:
            state.status = FormStatus.EXECUTING

            # Execute form action
            result = await form.execute(state.filled_slots)

            state.status = FormStatus.EXECUTED
            state.execution_result = result

            return state, result

        except Exception as e:
            state.status = FormStatus.FAILED
            state.error_message = str(e)
            return state, {
                "success": False,
                "message": f"ত্রুটি: {str(e)}"
            }

    def _get_next_slot(
        self,
        form: BaseForm,
        state: FormState
    ) -> Optional[BaseSlot]:
        """Get next unfilled slot."""
        slots = form.required_slots()
        for slot in slots:
            if slot.name not in state.filled_slots:
                return slot
        return None

    def cancel_form(self, form: BaseForm, state: FormState) -> tuple[FormState, str]:
        """
        Cancel an active form.

        Args:
            form: Form instance
            state: Current form state

        Returns:
            tuple: (updated_state, cancellation_message)
        """
        state.status = FormStatus.CANCELLED
        state.current_slot = None
        msg = form.on_form_cancel()
        return state, msg or "ফর্ম বাতিল করা হয়েছে।"

