"""NID Status Check Form."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from ..base_form import BaseForm, BaseSlot
from ..adapters import NIDServiceAdapter, MockNIDServiceAdapter


class NIDStatusCheckForm(BaseForm):
    """Form to check NID application status."""

    def __init__(self, service_adapter: Optional[NIDServiceAdapter] = None):
        self.service_adapter = service_adapter or MockNIDServiceAdapter()

    @property
    def form_name(self) -> str:
        return "nid_status_check"

    @property
    def trigger_tags(self) -> List[str]:
        """Tags that trigger NID status check form."""
        return [
            "correction_application_status",
            "correction_application_current_status",
            "smart_card_status",
            "smart_card_printing_status_sms_how",
            "nrb_application_status"
        ]

    def required_slots(self) -> List[BaseSlot]:
        return [
            BaseSlot(
                name="nid_number",
                prompt="আপনার NID নম্বর প্রদান করুন (17 সংখ্যা):",
                slot_type="numeric",
                validation_regex=r"^\d{17}$",
                error_message="NID নম্বর অবশ্যই 17 সংখ্যার হতে হবে। অনুগ্রহ করে আবার চেষ্টা করুন।",
                max_retries=3
            ),
            BaseSlot(
                name="date_of_birth",
                prompt="আপনার জন্ম তারিখ প্রদান করুন (DD-MM-YYYY):",
                slot_type="date",
                validation_func=self._validate_date,
                error_message="সঠিক তারিখ ফরম্যাটে প্রদান করুন (DD-MM-YYYY), যেমন: 15-01-1990",
                max_retries=3
            )
        ]

    def _validate_date(self, date_str: str) -> bool:
        """Validate date format."""
        try:
            datetime.strptime(date_str, "%d-%m-%Y")
            return True
        except (ValueError, TypeError):
            return False

    async def execute(self, filled_slots: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NID status check."""
        nid_number = filled_slots.get("nid_number")
        dob = filled_slots.get("date_of_birth")

        service_result = await self.service_adapter.check_status(
            nid_number=nid_number,
            date_of_birth=dob
        )

        status = service_result.get("status", "অজানা (Unknown)")

        return {
            "success": True,
            "status": status,
            "message": f"আপনার NID {nid_number} এর স্ট্যাটাস: {status}",
            "nid_number": nid_number,
            "date_of_birth": dob,
            "raw_result": service_result
        }

