#!/usr/bin/env python3
"""
Proof of Concept: Credit Card Service Form
Demonstrates extensibility of current form implementation.
"""

import sys
from pathlib import Path

# Project root - go up from examples/ to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import List
from datetime import datetime
from src.application.forms.base_form import BaseForm, BaseSlot


class CreditCardVerificationForm(BaseForm):
    """
    Security verification form for credit card service.

    This uses the CURRENT BaseForm implementation without modifications.
    Demonstrates that security verification works out-of-the-box.
    """

    @property
    def form_name(self) -> str:
        return "credit_card_verification"

    @property
    def trigger_tags(self) -> List[str]:
        return [
            "credit_card_balance_inquiry",
            "credit_card_service",
            "card_customer_service"
        ]

    @property
    def slots(self) -> List[BaseSlot]:
        return [
            BaseSlot(
                name="full_name",
                prompt="For security, may I have your full name as it appears on the card?",
                slot_type="text",
                validation_func=self._validate_name,
                error_message="Please provide your full name (minimum 3 characters)",
                max_retries=3
            ),
            BaseSlot(
                name="card_last4",
                prompt="Thank you. Please provide the last four digits of your card:",
                slot_type="numeric",
                validation_regex=r"^\d{4}$",
                error_message="Please provide exactly 4 digits",
                max_retries=3
            ),
            BaseSlot(
                name="otp",
                prompt="I've sent a one-time password to your registered mobile. Please read it out when you receive it:",
                slot_type="numeric",
                validation_regex=r"^\d{6}$",
                error_message="OTP must be exactly 6 digits",
                max_retries=3
            )
        ]

    def _validate_name(self, name: str) -> bool:
        """Validate name has minimum length."""
        return len(name.strip()) >= 3

    async def execute(self, slot_values: dict) -> dict:
        """
        Verify card security credentials.

        In production, this would call a real verification API.
        For demo, we mock the verification.
        """
        name = slot_values.get("full_name")
        card_last4 = slot_values.get("card_last4")
        otp = slot_values.get("otp")

        # Mock verification logic
        # In reality: await verify_with_bank_api(name, card_last4, otp)
        verified = self._mock_verify(name, card_last4, otp)

        if verified:
            # Mock account data
            account_data = self._mock_get_account_data(card_last4)

            return {
                "success": True,
                "verified": True,
                "message": f"Verified. Thank you, {name}. I have your Horizon Platinum card ending {card_last4}. How can I help you today?",
                "session_metadata": {
                    "authenticated": True,
                    "card_id": card_last4,
                    "customer_name": name,
                    "account_data": account_data
                }
            }
        else:
            return {
                "success": False,
                "verified": False,
                "message": "Verification failed. Please check your credentials and try again."
            }

    def _mock_verify(self, name: str, card_last4: str, otp: str) -> bool:
        """Mock verification - in production, call real API."""
        # Accept any 6-digit OTP for demo
        return len(otp) == 6

    def _mock_get_account_data(self, card_last4: str) -> dict:
        """Mock account data retrieval."""
        # In production: await fetch_from_bank_api(card_last4)
        return {
            "balance": 57840.23,
            "available_credit": 142159.77,
            "credit_limit": 200000.00,
            "statement_date": "2025-10-20",
            "due_date": "2025-11-09",
            "minimum_due": 4629.00,
            "last_payment": {
                "amount": 20000.00,
                "date": "2025-10-12",
                "method": "Mobile Banking"
            },
            "recent_transactions": [
                {"date": "2025-10-25", "merchant": "ElectroMart Dhanmondi", "amount": 18900.00},
                {"date": "2025-10-27", "merchant": "AirFly Tickets", "amount": 7500.00},
                {"date": "2025-10-28", "merchant": "Restaurant Dining", "amount": 1200.00},
                {"date": "2025-10-29", "merchant": "Fuel Station", "amount": 650.00}
            ]
        }


# ============================================================================
# Demonstration: How to handle post-verification queries
# ============================================================================

def handle_authenticated_query(query: str, session_data: dict) -> dict:
    """
    Handle queries after successful authentication.

    This demonstrates how to use dialogue policy + backend data
    to answer user questions WITHOUT needing a new form.

    In production, this would be part of the dialogue manager.
    """
    account_data = session_data.get("account_data", {})
    customer_name = session_data.get("customer_name", "Customer")

    query_lower = query.lower()

    # Balance inquiry
    if any(word in query_lower for word in ["balance", "outstanding", "how much"]):
        balance = account_data.get("balance", 0)
        available = account_data.get("available_credit", 0)
        limit = account_data.get("credit_limit", 0)

        return {
            "response": f"As of today, your outstanding balance is BDT {balance:,.2f}. "
                       f"Your available credit is BDT {available:,.2f} "
                       f"and your total credit limit is BDT {limit:,.2f}.",
            "action": "balance_inquiry"
        }

    # Minimum due inquiry
    elif any(word in query_lower for word in ["minimum", "due date", "payment due"]):
        min_due = account_data.get("minimum_due", 0)
        due_date = account_data.get("due_date", "")
        statement_date = account_data.get("statement_date", "")

        return {
            "response": f"Your statement generated on {statement_date}. "
                       f"The payment due date is {due_date}. "
                       f"The minimum amount due is BDT {min_due:,.2f}. "
                       f"If you pay the total outstanding by the due date, you won't be charged interest on eligible purchases.",
            "action": "payment_details"
        }

    # Last payment inquiry
    elif any(word in query_lower for word in ["last payment", "recent payment", "payment posted"]):
        last_payment = account_data.get("last_payment", {})
        amount = last_payment.get("amount", 0)
        date = last_payment.get("date", "")
        method = last_payment.get("method", "")

        return {
            "response": f"A payment of BDT {amount:,.2f} was posted on {date} via {method}. "
                       f"Would you like me to text or email a copy of your latest statement?",
            "action": "last_payment_inquiry"
        }

    # Transaction breakdown
    elif any(word in query_lower for word in ["transaction", "purchase", "what's driving", "breakdown"]):
        transactions = account_data.get("recent_transactions", [])

        if transactions:
            txn_list = "\n".join([
                f"• BDT {t['amount']:,.2f} at {t['merchant']} on {t['date']}"
                for t in transactions
            ])

            return {
                "response": f"Recent transactions:\n{txn_list}\n\nDoes that look familiar?",
                "action": "transaction_breakdown"
            }

    # Email statement (would trigger mini-form for confirmation)
    elif any(word in query_lower for word in ["email", "send statement"]):
        return {
            "response": "I can email your latest statement. Should I send it to your registered email address? (yes/no)",
            "action": "email_statement_request",
            "requires_confirmation": True
        }

    # Auto-payment setup (would trigger new form)
    elif any(word in query_lower for word in ["auto", "automatic payment", "set up payment"]):
        return {
            "response": "I can help set up automatic payment. Would you like to auto-debit:\n"
                       "1. Minimum due\n"
                       "2. Total due\n"
                       "3. Fixed amount\n\n"
                       "Please choose 1, 2, or 3:",
            "action": "auto_payment_setup",
            "triggers_new_form": "auto_payment_form"
        }

    # Payment link
    elif any(word in query_lower for word in ["payment link", "send link", "pay now"]):
        return {
            "response": "I'll send a payment link to your registered mobile number. "
                       "The link is valid for 24 hours and supports instant bank transfer or card-to-card payment.",
            "action": "send_payment_link",
            "action_executed": True
        }

    # Default
    else:
        return {
            "response": "I can help you with balance inquiries, payment details, transaction history, "
                       "statement requests, or payment setup. What would you like to know?",
            "action": "fallback"
        }


# ============================================================================
# Optional: Mini-forms for specific actions
# ============================================================================

class EmailStatementForm(BaseForm):
    """Simple confirmation form for emailing statement."""

    @property
    def form_name(self) -> str:
        return "email_statement"

    @property
    def trigger_tags(self) -> List[str]:
        return ["email_statement_request"]

    @property
    def slots(self) -> List[BaseSlot]:
        return [
            BaseSlot(
                name="confirm",
                prompt="Send statement to your registered email? (yes/no):",
                slot_type="text",
                validation_func=lambda x: x.lower() in ["yes", "no", "y", "n"],
                error_message="Please answer 'yes' or 'no'",
                max_retries=3
            )
        ]

    async def execute(self, slot_values: dict) -> dict:
        if slot_values["confirm"].lower() in ["yes", "y"]:
            # Mock: Send email
            return {
                "success": True,
                "message": "Done. I've sent the PDF to your registered email address. "
                          "Please check your inbox and spam folder."
            }
        else:
            return {
                "success": True,
                "message": "Okay, I won't send the statement. Is there anything else I can help with?"
            }


class AutoPaymentSetupForm(BaseForm):
    """Form for setting up automatic payments."""

    @property
    def form_name(self) -> str:
        return "auto_payment_setup"

    @property
    def trigger_tags(self) -> List[str]:
        return ["auto_payment_request"]

    @property
    def slots(self) -> List[BaseSlot]:
        return [
            BaseSlot(
                name="payment_type",
                prompt="Choose payment type:\n1. Minimum due\n2. Total due\n3. Fixed amount\n\nEnter 1, 2, or 3:",
                slot_type="text",
                validation_func=lambda x: x in ["1", "2", "3"],
                error_message="Please choose 1, 2, or 3",
                max_retries=3
            ),
            BaseSlot(
                name="source_account",
                prompt="Which account should we debit from? (e.g., Horizon Savings):",
                slot_type="text",
                validation_func=lambda x: len(x.strip()) >= 3,
                error_message="Please provide a valid account name",
                max_retries=3
            )
        ]

    async def execute(self, slot_values: dict) -> dict:
        payment_types = {
            "1": "Minimum due",
            "2": "Total due",
            "3": "Fixed amount"
        }

        payment_type = payment_types.get(slot_values["payment_type"], "Unknown")
        account = slot_values["source_account"]

        # Mock: Configure auto-payment
        return {
            "success": True,
            "message": f"Auto-debit enabled for '{payment_type}' from your {account}. "
                      f"It will take effect from the next cycle. "
                      f"For this cycle, you'll still need to make a manual payment by the due date."
        }


# ============================================================================
# Demo/Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CREDIT CARD SERVICE FORM - PROOF OF CONCEPT")
    print("=" * 80)
    print("\nThis demonstrates:")
    print("✅ Security verification using current BaseForm")
    print("✅ Post-auth queries via dialogue policy routing")
    print("✅ Mini-forms for specific actions (email, auto-pay)")
    print("\n" + "=" * 80)
    print("\n")

    # Simulate the conversation flow
    print("📞 SIMULATED CONVERSATION\n")

    # Step 1: User triggers credit card service
    print("User: I want to check my credit card balance")
    print("→ Triggers: CreditCardVerificationForm\n")

    # Step 2: Security verification (using current BaseForm!)
    print("Bot: For security, may I have your full name as it appears on the card?")
    print("User: Md. Arif Rahman\n")

    print("Bot: Thank you. Please provide the last four digits of your card:")
    print("User: 2749\n")

    print("Bot: I've sent a one-time password to your registered mobile.")
    print("User: 482119\n")

    # Step 3: Verification complete, session authenticated
    verification_form = CreditCardVerificationForm()
    slot_values = {
        "full_name": "Md. Arif Rahman",
        "card_last4": "2749",
        "otp": "482119"
    }

    import asyncio
    result = asyncio.run(verification_form.execute(slot_values))

    print(f"Bot: {result['message']}\n")
    print("✅ Security verification complete!")
    print(f"✅ Session marked as authenticated: {result['session_metadata']['authenticated']}\n")
    print("=" * 80)
    print("\n📊 POST-AUTHENTICATION QUERIES (Using Dialogue Policy)\n")

    session_data = result["session_metadata"]

    # Query 1: Balance
    print("User: What's my balance?")
    response = handle_authenticated_query("What's my balance?", session_data)
    print(f"Bot: {response['response']}\n")

    # Query 2: Minimum due
    print("User: What's the minimum due and due date?")
    response = handle_authenticated_query("What's the minimum due?", session_data)
    print(f"Bot: {response['response']}\n")

    # Query 3: Last payment
    print("User: When was my last payment posted?")
    response = handle_authenticated_query("last payment", session_data)
    print(f"Bot: {response['response']}\n")

    # Query 4: Transactions
    print("User: The balance looks high. What's driving it up?")
    response = handle_authenticated_query("breakdown transactions", session_data)
    print(f"Bot: {response['response']}\n")

    # Query 5: Email statement (triggers mini-form)
    print("User: Email it, please")
    response = handle_authenticated_query("email statement", session_data)
    print(f"Bot: {response['response']}")
    print("→ Would trigger EmailStatementForm for confirmation\n")

    # Query 6: Auto-payment (triggers new form)
    print("User: Set up automatic payment")
    response = handle_authenticated_query("automatic payment", session_data)
    print(f"Bot: {response['response']}")
    print("→ Would trigger AutoPaymentSetupForm\n")

    print("=" * 80)
    print("\n✅ VERDICT: Current implementation CAN handle this!")
    print("\nApproach:")
    print("1. ✅ Security → Use CreditCardVerificationForm (current BaseForm)")
    print("2. ✅ Queries → Use dialogue policy with backend data")
    print("3. ✅ Actions → Use mini-forms (EmailStatementForm, AutoPaymentForm)")
    print("\nNo changes to core needed! Just smart composition. 🎯")
    print("=" * 80)
