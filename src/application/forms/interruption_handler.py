"""Form interruption handling for FAQ questions during forms."""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class InterruptionState:
    """State for form interruption."""

    awaiting_choice: bool = False
    paused_form_name: Optional[str] = None
    paused_slot: Optional[str] = None
    interrupting_query: Optional[str] = None
    interruption_count: int = 0


class FormInterruptionHandler:
    """
    Handles interruptions during form slot collection.

    When user asks FAQ question mid-form:
    1. Detect interruption (different tag)
    2. Ask: continue form or handle new query?
    3. Route accordingly
    """

    def __init__(self, max_interruptions: int = 3):
        """
        Initialize interruption handler.

        Args:
            max_interruptions: Max FAQs allowed during form (default: 3)
        """
        self.max_interruptions = max_interruptions

    def detect_interruption(
        self,
        current_form_tags: list,
        nlp_result_tags: list
    ) -> bool:
        """
        Detect if user's query is interrupting the active form.

        Args:
            current_form_tags: Tags associated with active form
            nlp_result_tags: Tags from current NLP result

        Returns:
            True if interruption detected
        """
        if not current_form_tags or not nlp_result_tags:
            return False

        # Interruption = different tag set
        current_tags_set = set(current_form_tags)
        nlp_tags_set = set(nlp_result_tags)

        return not current_tags_set.intersection(nlp_tags_set)

    def should_allow_interruption(
        self,
        interruption_state: InterruptionState
    ) -> tuple[bool, Optional[str]]:
        """
        Check if interruption should be allowed.

        Args:
            interruption_state: Current interruption state

        Returns:
            tuple: (allow, reason_if_not_allowed)
        """
        if interruption_state.interruption_count >= self.max_interruptions:
            return False, (
                f"সর্বাধিক {self.max_interruptions} বার বিঘ্ন অনুমোদিত। "
                "অনুগ্রহ করে বর্তমান ফর্ম সম্পন্ন করুন।"
            )

        return True, None

    def create_choice_prompt(
        self,
        form_name: str,
        current_slot: str
    ) -> str:
        """
        Create prompt asking user to choose.

        Args:
            form_name: Name of active form
            current_slot: Current slot being collected

        Returns:
            Choice prompt message
        """
        return (
            f"আপনি '{form_name}' ফর্ম পূরণ করছেন। "
            f"আপনি কি ফর্ম চালিয়ে যেতে চান, নাকি নতুন প্রশ্ন করতে চান?\n"
            "উত্তর দিন: 'চালিয়ে যান' অথবা 'নতুন প্রশ্ন'"
        )

    def parse_user_choice(self, user_input: str) -> Optional[str]:
        """
        Parse user's choice from input.

        Args:
            user_input: User's response

        Returns:
            "continue_form" | "handle_new" | None
        """
        user_input_lower = user_input.lower().strip()

        # Continue form keywords
        continue_keywords = [
            'চালিয়ে যান', 'চালিয়ে', 'continue', 'yes', 'হ্যাঁ',
            'ফর্ম', 'form', 'আগে'
        ]

        # New query keywords
        new_keywords = [
            'নতুন', 'প্রশ্ন', 'new', 'switch', 'no', 'না',
            'অন্য', 'different'
        ]

        if any(kw in user_input_lower for kw in continue_keywords):
            return "continue_form"

        if any(kw in user_input_lower for kw in new_keywords):
            return "handle_new"

        return None  # Unclear, ask again

    def increment_interruption(
        self,
        interruption_state: InterruptionState
    ) -> InterruptionState:
        """Increment interruption counter."""
        interruption_state.interruption_count += 1
        return interruption_state

    def reset_interruption_state(self) -> InterruptionState:
        """Reset interruption state."""
        return InterruptionState()

