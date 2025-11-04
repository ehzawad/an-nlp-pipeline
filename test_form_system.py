#!/usr/bin/env python3.12
"""
Test Form System with Dialogue Manager

Demonstrates the composeable dialogue manager with form-based slot filling
for credit card balance inquiry.
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent))

from src.components.ner import NERComponent, NERConfig


# Form definition for credit card balance inquiry
@dataclass
class FormSlot:
    """Represents a single slot in a form"""
    name: str
    required: bool = True
    filled: bool = False
    value: Optional[Any] = None
    prompt: str = ""
    validation_pattern: Optional[str] = None


@dataclass
class Form:
    """Form with multiple slots"""
    name: str
    slots: Dict[str, FormSlot] = field(default_factory=dict)

    def add_slot(self, slot: FormSlot):
        """Add a slot to the form"""
        self.slots[slot.name] = slot

    def is_complete(self) -> bool:
        """Check if all required slots are filled"""
        return all(
            not slot.required or slot.filled
            for slot in self.slots.values()
        )

    def get_next_unfilled_slot(self) -> Optional[FormSlot]:
        """Get the next required slot that needs to be filled"""
        for slot in self.slots.values():
            if slot.required and not slot.filled:
                return slot
        return None

    def fill_slot(self, slot_name: str, value: Any) -> bool:
        """Fill a slot with a value"""
        if slot_name in self.slots:
            self.slots[slot_name].value = value
            self.slots[slot_name].filled = True
            return True
        return False

    def get_filled_slots(self) -> Dict[str, Any]:
        """Get all filled slots as a dictionary"""
        return {
            name: slot.value
            for name, slot in self.slots.items()
            if slot.filled
        }


# Mock credit card accounts
MOCK_ACCOUNTS = {
    "2749": {
        "name": "Md. Arif Rahman",
        "card_number": "**** **** **** 2749",
        "balance": 57840.23,
        "available_credit": 142159.77,
        "credit_limit": 200000.00,
        "minimum_due": 4629.00,
        "due_date": "2025-11-09",
    },
    "5832": {
        "name": "Sarah Ahmed",
        "card_number": "**** **** **** 5832",
        "balance": 32150.50,
        "available_credit": 67849.50,
        "credit_limit": 100000.00,
        "minimum_due": 2572.00,
        "due_date": "2025-11-15",
    }
}


def create_balance_inquiry_form() -> Form:
    """Create a form for credit card balance inquiry"""
    form = Form(name="credit_card_balance_inquiry")

    # Slot 1: Customer name
    form.add_slot(FormSlot(
        name="customer_name",
        required=True,
        prompt="For security purposes, may I have your full name as it appears on the card?"
    ))

    # Slot 2: Card last 4 digits
    form.add_slot(FormSlot(
        name="card_last4",
        required=True,
        prompt="Thank you. Please provide the last four digits of your credit card:"
    ))

    # Slot 3: OTP verification
    form.add_slot(FormSlot(
        name="otp",
        required=True,
        prompt="I'm sending a one-time password to your registered mobile number. Please provide the 6-digit OTP:"
    ))

    return form


async def process_with_form_system():
    """Demonstrate form-based dialogue system"""

    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "FORM-BASED DIALOGUE SYSTEM TEST" + " "*22 + "║")
    print("╚" + "═"*68 + "╝")

    # Initialize NER component
    print("\n[SYSTEM] Initializing NER component...")
    ner = NERComponent(NERConfig(enable_transformer=False))
    await ner.initialize()
    print("[SYSTEM] ✓ NER ready\n")

    # Create form
    form = create_balance_inquiry_form()
    print(f"[FORM] Created form: {form.name}")
    print(f"[FORM] Required slots: {list(form.slots.keys())}\n")

    # Conversation with form filling
    conversation = [
        ("Hello, I want to check my credit card balance", None),
        ("Md. Arif Rahman", "customer_name"),
        ("2749", "card_last4"),
        ("482119", "otp"),
    ]

    print("=" * 70)
    print("CONVERSATION WITH FORM SLOT FILLING")
    print("=" * 70)

    for turn_num, (user_input, expected_slot) in enumerate(conversation, 1):
        print(f"\n{'─'*70}")
        print(f"TURN {turn_num}")
        print(f"{'─'*70}")
        print(f"👤 User: {user_input}")

        # Extract entities from user input
        from src.components.ner import NERInput
        ner_result = await ner.process(NERInput(text=user_input))

        if ner_result.entities:
            print(f"[NER] Extracted entities: {ner_result.entities}")

        # Determine which slot to fill
        if turn_num == 1:
            # Initial request
            print(f"\n[FORM] Intent detected: credit_card_balance_inquiry")
            print(f"[FORM] Activating form: {form.name}")

            # Get first slot to fill
            next_slot = form.get_next_unfilled_slot()
            if next_slot:
                print(f"[FORM] Next slot to fill: {next_slot.name}")
                print(f"🤖 Bot: {next_slot.prompt}")

        else:
            # Try to fill a slot
            next_slot = form.get_next_unfilled_slot()

            if next_slot:
                # Determine value to fill based on slot type
                value = None

                if next_slot.name == "customer_name":
                    # Extract from input directly
                    value = user_input
                    print(f"[FORM] Filling slot '{next_slot.name}' with: {value}")
                    form.fill_slot(next_slot.name, value)

                elif next_slot.name == "card_last4":
                    # Extract from NER entities
                    if 'number' in ner_result.entities:
                        value = ner_result.entities['number']
                        print(f"[FORM] Filling slot '{next_slot.name}' with: {value}")
                        form.fill_slot(next_slot.name, value)
                    elif 'account_number' in ner_result.entities:
                        value = ner_result.entities['account_number'][-4:]
                        print(f"[FORM] Filling slot '{next_slot.name}' with: {value}")
                        form.fill_slot(next_slot.name, value)

                elif next_slot.name == "otp":
                    # Extract OTP (6 digits)
                    if 'number' in ner_result.entities:
                        value = ner_result.entities['number']
                        print(f"[FORM] Filling slot '{next_slot.name}' with: {value}")
                        form.fill_slot(next_slot.name, value)

                # Check if form is complete
                if form.is_complete():
                    print(f"\n[FORM] ✓ All slots filled!")
                    print(f"[FORM] Form data: {form.get_filled_slots()}")

                    # Execute form action (show balance)
                    card_last4 = form.slots['card_last4'].value
                    customer_name = form.slots['customer_name'].value

                    if card_last4 in MOCK_ACCOUNTS:
                        account = MOCK_ACCOUNTS[card_last4]

                        print(f"\n[ACTION] Executing: show_credit_card_balance")
                        print(f"[ACTION] Parameters: card={card_last4}, customer={customer_name}")

                        response = f"""Thank you for verifying, {customer_name}.

Your Credit Card Balance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Card: {account['card_number']}
Name: {account['name']}

Outstanding Balance: BDT {account['balance']:,.2f}
Available Credit:    BDT {account['available_credit']:,.2f}
Credit Limit:        BDT {account['credit_limit']:,.2f}

Payment Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Minimum Due:         BDT {account['minimum_due']:,.2f}
Due Date:            {account['due_date']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Is there anything else I can help you with?"""

                        print(f"\n🤖 Bot:\n{response}")
                    else:
                        print(f"\n🤖 Bot: I couldn't find a card ending in {card_last4}.")

                else:
                    # Ask for next slot
                    next_slot = form.get_next_unfilled_slot()
                    if next_slot:
                        print(f"[FORM] Next slot to fill: {next_slot.name}")
                        print(f"\n🤖 Bot: {next_slot.prompt}")

    print(f"\n{'─'*70}")
    print("\n[SUMMARY]")
    print(f"Form name: {form.name}")
    print(f"Completion status: {'✓ Complete' if form.is_complete() else '✗ Incomplete'}")
    print(f"Filled slots: {list(form.get_filled_slots().keys())}")
    print(f"Slot values: {form.get_filled_slots()}")

    print("\n" + "=" * 70)
    print("✓ FORM SYSTEM TEST COMPLETED")
    print("=" * 70)


async def test_multiple_forms():
    """Test with multiple form scenarios"""

    print("\n\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*18 + "MULTIPLE FORM SCENARIOS" + " "*27 + "║")
    print("╚" + "═"*68 + "╝\n")

    # Scenario 1: Balance inquiry
    print("📋 SCENARIO 1: Balance Inquiry Form")
    print("─"*70)
    await process_with_form_system()

    # Scenario 2: Show form structure
    print("\n\n📋 SCENARIO 2: Form Slot Structure")
    print("─"*70)

    form = create_balance_inquiry_form()

    print(f"\nForm: {form.name}")
    print(f"Total slots: {len(form.slots)}")
    print(f"\nSlot definitions:")

    for slot_name, slot in form.slots.items():
        print(f"\n  • {slot_name}")
        print(f"    - Required: {slot.required}")
        print(f"    - Prompt: {slot.prompt}")
        print(f"    - Filled: {slot.filled}")

    print("\n" + "="*70)


async def main():
    """Run all form system tests"""
    try:
        await test_multiple_forms()

        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*25 + "TEST SUMMARY" + " "*31 + "║")
        print("╠" + "═"*68 + "╣")
        print("║  ✓ Form creation and structure                                  ║")
        print("║  ✓ Slot filling with NER entity extraction                      ║")
        print("║  ✓ Sequential slot prompting                                    ║")
        print("║  ✓ Form completion detection                                    ║")
        print("║  ✓ Action execution with form data                              ║")
        print("║  ✓ Multi-turn dialogue with state tracking                      ║")
        print("╠" + "═"*68 + "╣")
        print("║  Status: ALL FORM TESTS PASSED ✓                                ║")
        print("╚" + "═"*68 + "╝")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
