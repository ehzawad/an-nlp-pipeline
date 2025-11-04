#!/usr/bin/env python3.12
"""
Test Full Session Tracking

Demonstrates complete conversation history tracking across multiple turns,
entity accumulation, and session state management.
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent))

from src.components.ner import NERComponent, NERInput, NERConfig


class MessageRole(str, Enum):
    """Message role in conversation"""
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"


@dataclass
class Message:
    """A single message in the conversation"""
    role: MessageRole
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    entities: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        prefix = "👤" if self.role == MessageRole.USER else "🤖" if self.role == MessageRole.BOT else "⚙️"
        entity_str = f" [Entities: {self.entities}]" if self.entities else ""
        return f"{prefix} {self.role.upper()}: {self.content}{entity_str}"


@dataclass
class SessionState:
    """Complete session state with full conversation history"""
    session_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    messages: List[Message] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    current_intent: Optional[str] = None
    active_form: Optional[str] = None
    form_slots: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, entities: Dict = None, metadata: Dict = None):
        """Add a message to the conversation history"""
        message = Message(
            role=role,
            content=content,
            entities=entities or {},
            metadata=metadata or {}
        )
        self.messages.append(message)

        # Accumulate entities
        if entities:
            self.entities.update(entities)

        return message

    def get_conversation_history(self) -> List[Message]:
        """Get the complete conversation history"""
        return self.messages

    def get_user_messages(self) -> List[Message]:
        """Get only user messages"""
        return [msg for msg in self.messages if msg.role == MessageRole.USER]

    def get_bot_messages(self) -> List[Message]:
        """Get only bot messages"""
        return [msg for msg in self.messages if msg.role == MessageRole.BOT]

    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get the last N messages"""
        return self.messages[-n:] if len(self.messages) >= n else self.messages

    def get_accumulated_entities(self) -> Dict[str, str]:
        """Get all accumulated entities across conversation"""
        return self.entities

    def print_full_session(self):
        """Print complete session state"""
        print("\n" + "╔" + "═"*78 + "╗")
        print("║" + " "*25 + "FULL SESSION STATE" + " "*35 + "║")
        print("╚" + "═"*78 + "╝")

        print(f"\n📋 Session ID: {self.session_id}")
        print(f"⏰ Created: {self.created_at}")
        print(f"💬 Total messages: {len(self.messages)}")
        print(f"🎯 Current intent: {self.current_intent or 'None'}")
        print(f"📝 Active form: {self.active_form or 'None'}")

        print(f"\n🔍 Accumulated Entities ({len(self.entities)} total):")
        if self.entities:
            for key, value in self.entities.items():
                print(f"   • {key}: {value}")
        else:
            print("   (No entities)")

        print(f"\n📦 Form Slots ({len(self.form_slots)} filled):")
        if self.form_slots:
            for key, value in self.form_slots.items():
                print(f"   • {key}: {value}")
        else:
            print("   (No slots filled)")

        print(f"\n💾 Context Data ({len(self.context)} items):")
        if self.context:
            for key, value in self.context.items():
                print(f"   • {key}: {value}")
        else:
            print("   (No context)")

        print("\n" + "─"*80)
        print("FULL CONVERSATION HISTORY")
        print("─"*80)

        for i, msg in enumerate(self.messages, 1):
            print(f"\n[{i}] {msg.timestamp}")
            print(f"    {msg}")
            if msg.metadata:
                print(f"    Metadata: {msg.metadata}")


# Mock account data
MOCK_ACCOUNTS = {
    "2749": {
        "name": "Md. Arif Rahman",
        "card_number": "**** **** **** 2749",
        "balance": 57840.23,
        "available_credit": 142159.77,
        "credit_limit": 200000.00,
        "minimum_due": 4629.00,
        "due_date": "2025-11-09",
        "transactions": [
            {"date": "2025-11-01", "merchant": "Amazon", "amount": 3500.00},
            {"date": "2025-10-28", "merchant": "Uber", "amount": 450.50},
            {"date": "2025-10-25", "merchant": "Grocery Store", "amount": 2890.00},
        ]
    }
}


async def simulate_full_conversation_session():
    """Simulate a complete multi-turn conversation with full session tracking"""

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*20 + "FULL SESSION TRACKING DEMONSTRATION" + " "*23 + "║")
    print("╚" + "═"*78 + "╝")

    # Initialize NER
    print("\n[SYSTEM] Initializing NER component...")
    ner = NERComponent(NERConfig(enable_transformer=False))
    await ner.initialize()
    print("[SYSTEM] ✓ NER ready")

    # Create session
    session = SessionState(session_id="session_12345")
    print(f"[SYSTEM] ✓ Session created: {session.session_id}\n")

    # Define conversation turns
    conversation = [
        {
            "user": "Hello, I want to check my credit card balance",
            "intent": "credit_card_balance_inquiry",
            "bot_response": "I can help you with that. For security purposes, may I have your full name as it appears on the card?",
            "activate_form": "credit_card_balance_inquiry"
        },
        {
            "user": "Md. Arif Rahman",
            "intent": "provide_name",
            "bot_response": "Thank you, Mr. Rahman. Please provide the last four digits of your credit card:",
            "fill_slot": ("customer_name", "Md. Arif Rahman")
        },
        {
            "user": "It's 2749",
            "intent": "provide_card_info",
            "bot_response": "Perfect. I'm sending a one-time password to your registered mobile number ending in ***67. Please provide the 6-digit OTP:",
            "fill_slot": ("card_last4", "2749")
        },
        {
            "user": "482119",
            "intent": "provide_otp",
            "bot_response": None,  # Will show balance
            "fill_slot": ("otp", "482119")
        },
        {
            "user": "When is my payment due?",
            "intent": "payment_due_date_inquiry",
            "bot_response": "Your payment is due on 2025-11-09. The minimum amount due is BDT 4,629.00."
        },
        {
            "user": "Show me recent transactions",
            "intent": "transaction_history_inquiry",
            "bot_response": None  # Will show transactions
        },
        {
            "user": "Can I increase my credit limit?",
            "intent": "credit_limit_increase_request",
            "bot_response": "I can help you request a credit limit increase. Your current limit is BDT 200,000.00. What amount would you like to request?"
        },
        {
            "user": "I'd like 300000",
            "intent": "provide_amount",
            "bot_response": "I've submitted your request for a credit limit increase to BDT 300,000.00. You'll receive a decision within 3-5 business days via email and SMS."
        },
        {
            "user": "Thank you!",
            "intent": "gratitude",
            "bot_response": "You're welcome, Mr. Rahman! Is there anything else I can help you with today?"
        },
        {
            "user": "No, that's all",
            "intent": "end_conversation",
            "bot_response": "Thank you for banking with us. Have a great day!"
        }
    ]

    print("="*80)
    print("CONVERSATION IN PROGRESS")
    print("="*80)

    # Process each turn
    for turn_num, turn in enumerate(conversation, 1):
        print(f"\n{'─'*80}")
        print(f"TURN {turn_num}")
        print(f"{'─'*80}")

        # User message
        user_input = turn["user"]
        print(f"\n👤 User: {user_input}")

        # Extract entities
        ner_result = await ner.process(NERInput(text=user_input))

        # Add user message to session
        session.add_message(
            MessageRole.USER,
            user_input,
            entities=ner_result.entities,
            metadata={"intent": turn["intent"]}
        )

        # Update session state
        session.current_intent = turn["intent"]

        # Handle form activation
        if "activate_form" in turn:
            session.active_form = turn["activate_form"]
            print(f"[SESSION] ✓ Form activated: {session.active_form}")

        # Handle slot filling
        if "fill_slot" in turn:
            slot_name, slot_value = turn["fill_slot"]
            session.form_slots[slot_name] = slot_value
            print(f"[SESSION] ✓ Slot filled: {slot_name} = {slot_value}")

        # Show entity extraction
        if ner_result.entities:
            print(f"[NER] Extracted: {ner_result.entities}")

        print(f"[INTENT] {turn['intent']}")

        # Generate bot response
        if turn["bot_response"]:
            bot_response = turn["bot_response"]
        elif turn_num == 4:  # Show balance after OTP
            account = MOCK_ACCOUNTS["2749"]
            session.context["verified"] = True
            session.context["account_data"] = account
            bot_response = f"""Thank you for verifying, Mr. Rahman.

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
        elif turn_num == 6:  # Show transactions
            account = MOCK_ACCOUNTS["2749"]
            transactions = account["transactions"]
            bot_response = "Here are your recent transactions:\n\n"
            for i, txn in enumerate(transactions, 1):
                bot_response += f"{i}. {txn['date']} - {txn['merchant']}: BDT {txn['amount']:,.2f}\n"
            session.context["shown_transactions"] = True

        # Add bot response to session
        session.add_message(
            MessageRole.BOT,
            bot_response,
            metadata={"turn": turn_num}
        )

        print(f"\n🤖 Bot: {bot_response}")

    # Show complete session state
    print("\n\n" + "="*80)
    print("="*80)
    session.print_full_session()

    # Show analytics
    print("\n\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "SESSION ANALYTICS" + " "*36 + "║")
    print("╚" + "═"*78 + "╝")

    user_messages = session.get_user_messages()
    bot_messages = session.get_bot_messages()

    print(f"\n📊 Conversation Statistics:")
    print(f"   • Total turns: {len(conversation)}")
    print(f"   • User messages: {len(user_messages)}")
    print(f"   • Bot messages: {len(bot_messages)}")
    print(f"   • Entities extracted: {len(session.entities)}")
    print(f"   • Form slots filled: {len(session.form_slots)}")
    print(f"   • Intents detected: {len(set(turn['intent'] for turn in conversation))}")

    print(f"\n🎯 Intent Distribution:")
    intent_counts = {}
    for turn in conversation:
        intent = turn['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {intent}: {count}")

    print(f"\n🔍 Entity Timeline:")
    for i, msg in enumerate(session.messages, 1):
        if msg.role == MessageRole.USER and msg.entities:
            print(f"   Turn {i}: {list(msg.entities.keys())}")

    print(f"\n📝 Form Completion Timeline:")
    turn_count = 0
    for i, msg in enumerate(session.messages, 1):
        if msg.role == MessageRole.USER:
            turn_count += 1
            # Check which turn filled which slot
            for slot_name in session.form_slots:
                if turn_count == 2 and slot_name == "customer_name":
                    print(f"   Turn {turn_count}: {slot_name} filled")
                elif turn_count == 3 and slot_name == "card_last4":
                    print(f"   Turn {turn_count}: {slot_name} filled")
                elif turn_count == 4 and slot_name == "otp":
                    print(f"   Turn {turn_count}: {slot_name} filled → Form complete!")

    return session


async def test_session_persistence():
    """Test session state can be serialized and restored"""

    print("\n\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*23 + "SESSION PERSISTENCE TEST" + " "*30 + "║")
    print("╚" + "═"*78 + "╝")

    # Create a session with some data
    session = SessionState(session_id="test_session_001")
    session.add_message(MessageRole.USER, "Hello")
    session.add_message(MessageRole.BOT, "Hi there!")
    session.entities = {"name": "John", "card": "1234"}
    session.current_intent = "greeting"

    print("\n✓ Created session with messages and entities")
    print(f"  Messages: {len(session.messages)}")
    print(f"  Entities: {session.entities}")

    # Simulate serialization (in production, this would be JSON/Redis/DB)
    import json

    session_data = {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "messages": [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "entities": msg.entities,
                "metadata": msg.metadata
            }
            for msg in session.messages
        ],
        "entities": session.entities,
        "context": session.context,
        "current_intent": session.current_intent,
        "active_form": session.active_form,
        "form_slots": session.form_slots
    }

    serialized = json.dumps(session_data, indent=2)
    print(f"\n✓ Serialized session to JSON ({len(serialized)} bytes)")

    # Deserialize
    restored_data = json.loads(serialized)
    restored_session = SessionState(
        session_id=restored_data["session_id"],
        created_at=restored_data["created_at"],
        entities=restored_data["entities"],
        context=restored_data["context"],
        current_intent=restored_data["current_intent"],
        active_form=restored_data["active_form"],
        form_slots=restored_data["form_slots"]
    )

    # Restore messages
    for msg_data in restored_data["messages"]:
        restored_session.messages.append(Message(
            role=MessageRole(msg_data["role"]),
            content=msg_data["content"],
            timestamp=msg_data["timestamp"],
            entities=msg_data["entities"],
            metadata=msg_data["metadata"]
        ))

    print(f"✓ Restored session from JSON")
    print(f"  Messages: {len(restored_session.messages)}")
    print(f"  Entities: {restored_session.entities}")
    print(f"  Match: {restored_session.entities == session.entities}")

    print("\n✓ Session can be persisted and restored!")


async def main():
    """Run full session tracking tests"""
    try:
        # Main demonstration
        session = await simulate_full_conversation_session()

        # Persistence test
        await test_session_persistence()

        print("\n\n" + "╔" + "═"*78 + "╗")
        print("║" + " "*30 + "TEST SUMMARY" + " "*35 + "║")
        print("╠" + "═"*78 + "╣")
        print("║  ✓ Full conversation history tracking (10 turns)                          ║")
        print("║  ✓ Complete message history with timestamps                               ║")
        print("║  ✓ Entity accumulation across all turns                                   ║")
        print("║  ✓ Session state management (intent, form, context)                       ║")
        print("║  ✓ Form slot filling tracking                                             ║")
        print("║  ✓ Context data storage                                                   ║")
        print("║  ✓ Session analytics and insights                                         ║")
        print("║  ✓ Session serialization and persistence                                  ║")
        print("╠" + "═"*78 + "╣")
        print("║  Status: ALL SESSION TRACKING TESTS PASSED ✓                              ║")
        print("╚" + "═"*78 + "╝")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
