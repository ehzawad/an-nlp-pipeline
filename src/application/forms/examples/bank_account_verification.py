"""Bank account verification form with async API validation."""

import asyncio
from typing import List, Dict, Any
from ..base_form import BaseForm, BaseSlot


class BankAccountVerificationForm(BaseForm):
    """Form that verifies bank account ownership with async API validation."""

    @property
    def form_name(self) -> str:
        return "bank_account_verification"

    @property
    def trigger_tags(self) -> List[str]:
        return ["verify_account", "link_account", "validate_account", "check_account"]

    def required_slots(self) -> List[BaseSlot]:
        return [
            BaseSlot(
                name="account_number",
                prompt="Please enter your 10-16 digit account number:",
                validation_regex=r"^\d{10,16}$",
                validation_func=self._validate_account_async,  # ASYNC validation!
                error_message="Account number is invalid or doesn't exist in our system",
                max_retries=3
            ),
            BaseSlot(
                name="phone",
                prompt="Enter your registered phone number (11 digits):",
                validation_regex=r"^\d{11}$",
                error_message="Phone number must be exactly 11 digits"
            )
        ]

    async def _validate_account_async(self, account_number: str) -> bool:
        """
        Async validation function that calls external API.
        
        Demonstrates API call DURING slot collection.
        """
        # Simulate async API latency
        await asyncio.sleep(0.1)

        # Mock validation: accounts starting with 9 are valid
        is_valid = account_number.startswith('9')

        return is_valid

    async def execute(self, filled_slots: Dict[str, Any]) -> Dict[str, Any]:
        """Execute account linking after all slots collected and validated."""
        account_number = filled_slots["account_number"]
        phone = filled_slots["phone"]

        # In production: Call backend API to link account
        return {
            "success": True,
            "message": f"✅ Account {account_number} successfully verified and linked!\n"
                      f"Registered phone: {phone}\n"
                      f"You can now use this account for transactions.",
            "account_number": account_number,
            "phone": phone,
            "verification_id": f"VER-{account_number[-6:]}"
        }

