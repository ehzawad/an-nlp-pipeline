"""Base classes for form framework with async validation support."""

from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Any, Union, Awaitable, Dict
from abc import ABC, abstractmethod
import re
import inspect


class BaseSlot(BaseModel):
    """Slot definition with validation."""

    name: str = Field(..., description="Slot name/identifier")
    prompt: str = Field(..., description="What to ask the user")
    slot_type: str = Field(default="text", description="text|numeric|date|phone|email|etc")
    required: bool = Field(default=True, description="Is this slot required?")
    validation_regex: Optional[str] = Field(None, description="Regex pattern for validation")
    validation_func: Optional[Union[Callable[[Any], bool], Callable[[Any], Awaitable[bool]]]] = Field(
        None,
        description="Custom validation function (can be sync or async)"
    )
    error_message: Optional[str] = Field(None, description="Error message for validation failure")
    max_retries: int = Field(default=3, description="Max validation retry attempts")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        extra = "allow"

    async def validate_input(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate user input for this slot.

        Supports both synchronous and asynchronous validation functions.
        Async validation enables API calls during slot collection.

        Args:
            value: User input to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        # Type check for numeric
        if self.slot_type == "numeric":
            if not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    return False, self.error_message or "অনুগ্রহ করে একটি সঠিক সংখ্যা প্রদান করুন।"

        # Regex validation
        pattern = self.validation_regex or getattr(self, "validation_pattern", None)
        if pattern:
            if not re.match(pattern, str(value)):
                return False, self.error_message or f"{self.name} এর ফরম্যাট সঠিক নয়।"

        # Custom validation function (sync or async)
        if self.validation_func:
            try:
                if inspect.iscoroutinefunction(self.validation_func):
                    is_valid = await self.validation_func(value)
                else:
                    is_valid = self.validation_func(value)

                if not is_valid:
                    return False, self.error_message or f"{self.name} যাচাইকরণ ব্যর্থ হয়েছে।"
            except Exception as e:
                return False, str(e)

        return True, None


class BaseForm(ABC):
    """Abstract base class for forms."""

    @property
    @abstractmethod
    def form_name(self) -> str:
        """Unique form identifier."""
        pass

    @property
    @abstractmethod
    def trigger_tags(self) -> List[str]:
        """
        List of tags that trigger this form (OR logic - any tag triggers).

        Tags come from semantic search results.
        """
        pass

    @abstractmethod
    def required_slots(self) -> List[BaseSlot]:
        """Return list of required slots for this form."""
        pass

    async def validate_slot(self, slot_name: str, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate slot value.

        Args:
            slot_name: Name of the slot
            value: User input value

        Returns:
            tuple: (is_valid, error_message)
        """
        slots = {slot.name: slot for slot in self.required_slots()}
        if slot_name not in slots:
            return False, f"Unknown slot: {slot_name}"

        slot = slots[slot_name]
        return await slot.validate_input(value)

    @abstractmethod
    async def execute(self, filled_slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the form action with all filled slots.

        Args:
            filled_slots: Dictionary of slot_name -> value

        Returns:
            Result dictionary with:
            - success: bool
            - message: str
            - data: Optional[dict]
        """
        pass

    def get_slot_by_name(self, slot_name: str) -> Optional[BaseSlot]:
        """Get slot definition by name."""
        for slot in self.required_slots():
            if slot.name == slot_name:
                return slot
        return None

    def on_form_start(self) -> Optional[str]:
        """
        Hook called when form is activated.

        Returns:
            Optional welcome message
        """
        return None

    async def on_form_complete(self, filled_slots: Dict[str, Any]) -> Optional[str]:
        """
        Hook called before execute() when all slots are filled.

        Args:
            filled_slots: All collected slot values

        Returns:
            Optional confirmation message
        """
        return None

    def on_form_cancel(self) -> Optional[str]:
        """
        Hook called when form is cancelled.

        Returns:
            Optional cancellation message
        """
        return "ফর্ম বাতিল করা হয়েছে।"

