"""Service adapters for form backend integration."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import hashlib
import time


class NIDServiceAdapter(ABC):
    """Abstract adapter for NID service backend."""

    @abstractmethod
    async def check_status(self, nid_number: str, date_of_birth: str) -> Dict[str, Any]:
        """Check NID application status."""
        pass

    @abstractmethod
    async def submit_correction_request(
        self,
        nid_number: str,
        correction_type: str,
        details: str
    ) -> Dict[str, Any]:
        """Submit correction request."""
        pass


class MockNIDServiceAdapter(NIDServiceAdapter):
    """Mock implementation for testing."""

    async def check_status(self, nid_number: str, date_of_birth: str) -> Dict[str, Any]:
        """Mock status check."""
        # Simple mock logic based on last digit
        last_digit = int(nid_number[-1])

        status_map = {
            0: "প্রক্রিয়াধীন (Processing)",
            1: "অনুমোদিত (Approved)",
            2: "মুদ্রণে (Printing)",
            3: "বিতরণের জন্য প্রস্তুত (Ready for Delivery)",
            4: "প্রত্যাখ্যাত (Rejected)",
            5: "অতিরিক্ত তথ্য প্রয়োজন (Additional Info Required)",
            6: "যাচাইকরণ চলছে (Under Verification)",
            7: "সম্পন্ন (Completed)",
            8: "বিতরণ করা হয়েছে (Delivered)",
            9: "স্থগিত (On Hold)",
        }

        status = status_map.get(last_digit % 10, "অজানা (Unknown)")

        return {
            "success": True,
            "status": status,
            "nid_number": nid_number,
            "date_of_birth": date_of_birth,
            "application_date": "2025-10-01",
            "last_updated": "2025-10-30"
        }

    async def submit_correction_request(
        self,
        nid_number: str,
        correction_type: str,
        details: str
    ) -> Dict[str, Any]:
        """Mock correction request submission."""
        # Generate mock reference number
        timestamp = str(int(time.time()))
        hash_input = f"{nid_number}{timestamp}".encode()
        ref_num = hashlib.md5(hash_input).hexdigest()[:8].upper()
        reference_number = f"CORR-{ref_num}"

        return {
            "success": True,
            "reference_number": reference_number,
            "nid_number": nid_number,
            "correction_type": correction_type,
            "status": "জমা দেওয়া হয়েছে (Submitted)",
            "expected_completion_days": 10
        }

