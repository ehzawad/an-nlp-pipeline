"""Hotel Booking Form - Demo for testing form interruptions."""

from typing import List, Dict, Any
from datetime import datetime
import hashlib
import time
from ..base_form import BaseForm, BaseSlot


class HotelBookingForm(BaseForm):
    """Minimal hotel booking form to demonstrate form interruption handling."""

    @property
    def form_name(self) -> str:
        return "hotel_booking"

    @property
    def trigger_tags(self) -> List[str]:
        return [
            "hotel_booking",
            "room_reservation",
            "book_hotel_room"
        ]

    def required_slots(self) -> List[BaseSlot]:
        return [
            BaseSlot(
                name="city",
                prompt="Which city are you planning to stay in?",
                slot_type="text",
                validation_func=self._validate_city,
                error_message="Please provide a valid city name (minimum 2 characters)",
                max_retries=3
            ),
            BaseSlot(
                name="check_in_date",
                prompt="What's your check-in date? (Please use format DD-MM-YYYY, e.g., 14-11-2025)",
                slot_type="date",
                validation_func=self._validate_date,
                error_message="Please provide a valid date in DD-MM-YYYY format (e.g., 14-11-2025)",
                max_retries=3
            ),
            BaseSlot(
                name="num_guests",
                prompt="How many guests will be staying?",
                slot_type="numeric",
                validation_func=self._validate_guests,
                error_message="Please provide a number between 1 and 10",
                max_retries=3
            ),
            BaseSlot(
                name="full_name",
                prompt="To complete the booking, may I have your full name?",
                slot_type="text",
                validation_func=self._validate_name,
                error_message="Please provide your full name (minimum 3 characters)",
                max_retries=3
            )
        ]

    def _validate_city(self, city: str) -> bool:
        """Validate city name."""
        return len(city.strip()) >= 2

    def _validate_date(self, date_str: str) -> bool:
        """Validate date format and ensure it's not in the past."""
        try:
            date = datetime.strptime(date_str, "%d-%m-%Y")
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            return date >= today
        except (ValueError, TypeError):
            return False

    def _validate_guests(self, guests_str: str) -> bool:
        """Validate number of guests (1-10)."""
        try:
            num = int(guests_str)
            return 1 <= num <= 10
        except (ValueError, TypeError):
            return False

    def _validate_name(self, name: str) -> bool:
        """Validate full name."""
        return len(name.strip()) >= 3

    async def execute(self, filled_slots: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hotel booking (mock implementation)."""
        city = filled_slots.get("city")
        check_in = filled_slots.get("check_in_date")
        num_guests = filled_slots.get("num_guests")
        name = filled_slots.get("full_name")

        # Mock booking logic
        booking_ref = self._generate_booking_reference(name, check_in)
        total_amount = self._calculate_mock_amount(city, int(num_guests))

        return {
            "success": True,
            "booking_reference": booking_ref,
            "message": (
                f"Booking confirmed for {name}!\n\n"
                f"📍 City: {city}\n"
                f"📅 Check-in: {check_in}\n"
                f"👥 Guests: {num_guests}\n"
                f"🔖 Reference: {booking_ref}\n"
                f"💰 Estimated total: BDT {total_amount:,.2f}\n\n"
                f"A confirmation email has been sent. Looking forward to hosting you!"
            ),
            "city": city,
            "check_in_date": check_in,
            "num_guests": num_guests,
            "guest_name": name,
            "total_amount": total_amount
        }

    def _generate_booking_reference(self, name: str, date: str) -> str:
        """Generate mock booking reference."""
        timestamp = str(int(time.time()))
        hash_input = f"{name}{date}{timestamp}".encode()
        hash_value = hashlib.md5(hash_input).hexdigest()[:6].upper()
        return f"HB-{hash_value}"

    def _calculate_mock_amount(self, city: str, num_guests: int) -> float:
        """Calculate mock booking amount."""
        base_rates = {
            "dhaka": 8500,
            "chittagong": 7000,
            "sylhet": 6500,
            "cox's bazar": 9500,
        }
        base = base_rates.get(city.lower(), 7500)
        total = base + ((num_guests - 1) * 1500)
        total_with_tax = total * 1.15
        return total_with_tax

