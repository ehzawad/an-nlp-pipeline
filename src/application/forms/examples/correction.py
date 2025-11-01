"""NID Correction Request Form."""

from typing import List, Optional, Dict, Any
from ..base_form import BaseForm, BaseSlot
from ..adapters import NIDServiceAdapter, MockNIDServiceAdapter


class CorrectionRequestForm(BaseForm):
    """Form to request NID information correction."""

    def __init__(self, service_adapter: Optional[NIDServiceAdapter] = None):
        self.service_adapter = service_adapter or MockNIDServiceAdapter()

    @property
    def form_name(self) -> str:
        return "nid_correction_request"

    @property
    def trigger_tags(self) -> List[str]:
        """Tags that trigger NID correction request form."""
        return [
            "card_information_correction",
            "name_correction_in_nid_card",
            "age_and_dob_correction",
            "blood_group_correction",
            "educational_qualification_correction",
            "birthplace_correction_new",
            "nid_information_misentry_correction_procedure",
            "how_to_apply_for_correction_in_online_portal"
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
                name="correction_type",
                prompt="কী সংশোধন করতে চান? (নাম/ঠিকানা/ফটো/অন্যান্য):",
                slot_type="text",
                validation_func=self._validate_correction_type,
                error_message="অনুগ্রহ করে নাম, ঠিকানা, ফটো, অথবা অন্যান্য লিখুন।",
                max_retries=3
            ),
            BaseSlot(
                name="correction_details",
                prompt="সংশোধনের বিস্তারিত বর্ণনা করুন:",
                slot_type="text",
                validation_func=self._validate_details,
                error_message="অনুগ্রহ করে সংশোধনের বিস্তারিত তথ্য প্রদান করুন (কমপক্ষে 10 অক্ষর)।",
                max_retries=3
            )
        ]

    def _validate_correction_type(self, correction_type: str) -> bool:
        """Validate correction type."""
        valid_types = ["নাম", "ঠিকানা", "ফটো", "অন্যান্য", "name", "address", "photo", "other"]
        return correction_type.lower().strip() in [t.lower() for t in valid_types]

    def _validate_details(self, details: str) -> bool:
        """Validate correction details."""
        return len(details.strip()) >= 10

    async def execute(self, filled_slots: Dict[str, Any]) -> Dict[str, Any]:
        """Execute correction request submission."""
        nid_number = filled_slots.get("nid_number")
        correction_type = filled_slots.get("correction_type")
        details = filled_slots.get("correction_details")

        service_result = await self.service_adapter.submit_correction_request(
            nid_number=nid_number,
            correction_type=correction_type,
            details=details
        )

        reference_number = service_result.get("reference_number", "N/A")

        return {
            "success": True,
            "reference_number": reference_number,
            "message": (
                f"আপনার সংশোধন অনুরোধ জমা দেওয়া হয়েছে।\n\n"
                f"রেফারেন্স নম্বর: {reference_number}\n"
                f"NID: {nid_number}\n"
                f"সংশোধনের ধরন: {correction_type}\n\n"
                f"আপনার অনুরোধ পর্যালোচনা করা হবে এবং 7-10 কর্মদিবসের মধ্যে ফলাফল জানানো হবে।"
            ),
            "nid_number": nid_number,
            "correction_type": correction_type,
            "details": details,
            "raw_result": service_result
        }

