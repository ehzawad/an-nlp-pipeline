"""Integration tests for the EnhancedDialoguePipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.application.dialogue_service import EnhancedDialoguePipeline
from src.application.forms.form_registry import FormRegistry
from src.application.forms.examples.hotel_booking import HotelBookingForm
from src.application.forms.examples.nid_status import NIDStatusCheckForm
from src.application.forms.examples.correction import CorrectionRequestForm
from src.application.forms.examples.bank_account_verification import BankAccountVerificationForm
from src.application.policy.base_policy import Action, ActionType
from src.application.policy.policy_engine import PolicyEngine
from src.application.hooks.manager import HookManager, HookPoint
from src.application.session.models import SessionState
from src.application.session.manager import SessionManager
from src.application.session.store import InMemorySessionStore
from src.application.nlp_service import NLPResult, ClassificationResult, SearchResult


class FakeSessionStore(InMemorySessionStore):
    """In-memory session store extending InMemorySessionStore for tests."""
    
    def __init__(self):
        super().__init__(ttl_seconds=3600)  # Longer TTL for tests


class FakeSessionManager(SessionManager):
    def __init__(self):
        super().__init__(FakeSessionStore())


class FakeEventBus:
    def __init__(self):
        self.published: List[Any] = []

    async def publish(self, event):
        self.published.append(event)


class FakeEventStore:
    def __init__(self):
        self.events: List[Any] = []

    async def append(self, event):
        self.events.append(event)


class StaticSummarizer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.calls: List[str] = []

    async def summarize(self, text: str):
        self.calls.append(text)
        if not self.enabled:
            return text, {"summarized": False}
        return text[:100], {"summarized": True}


class StaticContextAugmenter:
    def __init__(self, prefix: Optional[str] = None):
        self.prefix = prefix
        self.calls: List[str] = []

    async def augment_if_needed(self, query: str, session_state):
        self.calls.append(query)
        if not self.prefix:
            return query, {}
        augmented = f"{self.prefix} {query}"
        return augmented, {"augmented": True, "context_tag": "test"}


class StaticNERExtractor:
    def __init__(self, entities: Optional[Dict[str, str]] = None):
        self.entities = entities or {}
        self.calls: List[str] = []

    async def extract(self, text: str, form_slots=None):
        self.calls.append(text)
        return {
            "entities": self.entities,
            "raw_entities": list(self.entities.items()),
            "method": "regex" if self.entities else "none",
        }


class ScriptedNLPService:
    def __init__(self, results: List[NLPResult]):
        self.results = results
        self.index = 0

    async def process(self, query: str, tenant_id: str, top_k: int = 5):
        if self.index < len(self.results):
            result = self.results[self.index]
            self.index += 1
            return result
        return self.results[-1]


class FakeTenantContext:
    def __init__(
        self,
        *,
        session_manager: FakeSessionManager,
        event_bus: FakeEventBus,
        event_store: FakeEventStore,
        hook_manager: HookManager,
        context_augmenter: StaticContextAugmenter,
        summarizer: StaticSummarizer,
        ner_extractor: StaticNERExtractor,
        form_registry: Optional[FormRegistry],
        nlp_service: ScriptedNLPService,
        policy_engine: PolicyEngine | FakePolicyEngine,
    ):
        self._session_manager = session_manager
        self._event_bus = event_bus
        self._event_store = event_store
        self._hook_manager = hook_manager
        self._context_augmenter = context_augmenter
        self._summarizer = summarizer
        self._ner_extractor = ner_extractor
        self._form_registry = form_registry
        self._nlp_service = nlp_service
        self._policy_engine = policy_engine

    async def get_session_manager(self):
        return self._session_manager

    async def get_event_bus(self):
        return self._event_bus

    async def get_event_store(self):
        return self._event_store

    async def get_hook_manager(self):
        return self._hook_manager

    async def get_context_augmenter(self):
        return self._context_augmenter

    async def get_summarizer(self):
        return self._summarizer

    async def get_ner_extractor(self):
        return self._ner_extractor

    async def get_form_registry(self):
        return self._form_registry

    async def get_nlp_service(self):
        return self._nlp_service

    async def get_policy_engine(self):
        return self._policy_engine


class FakePolicyEngine:
    def __init__(self, scripted_actions: List[Action]):
        self.scripted_actions = scripted_actions
        self.calls: List[Dict[str, Any]] = []

    async def decide_next_action(self, **kwargs):
        self.calls.append(kwargs)
        if self.scripted_actions:
            return self.scripted_actions.pop(0)
        return Action.fallback()


class PipelineHarness:
    def __init__(
        self,
        *,
        enable_forms: bool = True,
        enable_ner: bool = False,
        enable_context: bool = False,
        enable_summarization: bool = False,
        nlp_results: Optional[List[NLPResult]] = None,
        policy_engine: Optional[PolicyEngine | FakePolicyEngine] = None,
        context_prefix: Optional[str] = None,
        ner_entities: Optional[Dict[str, str]] = None,
    ):
        self.session_manager = FakeSessionManager()
        self.event_bus = FakeEventBus()
        self.event_store = FakeEventStore()
        self.hook_manager = HookManager()
        self.context_augmenter = StaticContextAugmenter(prefix=context_prefix)
        self.summarizer = StaticSummarizer(enabled=enable_summarization)
        self.ner_extractor = StaticNERExtractor(entities=ner_entities)
        self.form_registry = None

        if enable_forms:
            registry = FormRegistry()
            registry.register(HotelBookingForm())
            registry.register(NIDStatusCheckForm())
            registry.register(CorrectionRequestForm())
            registry.register(BankAccountVerificationForm())
            self.form_registry = registry

        self.policy_engine = policy_engine or PolicyEngine(
            enable_forms=enable_forms,
            form_registry=self.form_registry
        )

        self.nlp_service = ScriptedNLPService(nlp_results or [
            NLPResult(
                success=True,
                classification=[ClassificationResult(cluster="faq", confidence=0.9)],
                search_results=[SearchResult(question="Default", cluster="faq", tag=None, score=1.0)],
                strategy="single_cluster",
                confidence=0.9,
                processing_time_ms=5,
            )
        ])

        self.tenant_context = FakeTenantContext(
            session_manager=self.session_manager,
            event_bus=self.event_bus,
            event_store=self.event_store,
            hook_manager=self.hook_manager,
            context_augmenter=self.context_augmenter,
            summarizer=self.summarizer,
            ner_extractor=self.ner_extractor,
            form_registry=self.form_registry,
            nlp_service=self.nlp_service,
            policy_engine=self.policy_engine,
        )

        self.pipeline = EnhancedDialoguePipeline(
            enable_forms=enable_forms,
            enable_ner=enable_ner,
            enable_context_augmentation=enable_context,
            enable_summarization=enable_summarization,
        )

    async def run(self, session_id: str, query: str, correlation_id: str = "corr-1"):
        return await self.pipeline.process_turn(
            session_id=session_id,
            query=query,
            tenant_context=self.tenant_context,
            correlation_id=correlation_id,
        )


def make_faq_result(answer: str = "Answer") -> NLPResult:
    return NLPResult(
        success=True,
        classification=[ClassificationResult(cluster="faq", confidence=0.95)],
        search_results=[SearchResult(question=answer, cluster="faq", tag=None, score=0.99)],
        strategy="single_cluster",
        confidence=0.95,
        processing_time_ms=8,
    )


def make_form_trigger_result(tag: str) -> NLPResult:
    return NLPResult(
        success=True,
        classification=[ClassificationResult(cluster="forms", confidence=0.8)],
        search_results=[SearchResult(question="trigger", cluster="forms", tag=tag, score=0.9)],
        strategy="single_cluster",
        confidence=0.8,
        processing_time_ms=5,
    )


@pytest.mark.asyncio
async def test_single_turn_faq_response():
    harness = PipelineHarness(enable_forms=False, enable_ner=False, enable_context=False,
                              nlp_results=[make_faq_result("FAQ response")])

    result = await harness.run("session-1", "What is my status?")

    assert result["success"] is True
    assert result["text"] == "FAQ response"
    assert result["metadata"]["action_type"] == ActionType.FAQ_ANSWER.value
    assert len(harness.event_bus.published) >= 2
    session = await harness.session_manager.session_store.get("session-1")
    assert session is not None
    assert session.conversation_history[-1].bot_response == "FAQ response"


@pytest.mark.asyncio
async def test_multi_turn_form_completion():
    future_date = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%d-%m-%Y")
    nlp_results = [
        make_form_trigger_result("hotel_booking"),
        make_faq_result("irrelevant"),
        make_faq_result("irrelevant"),
        make_faq_result("irrelevant"),
        make_faq_result("irrelevant"),
    ]
    harness = PipelineHarness(enable_forms=True, enable_ner=False, enable_context=False,
                              nlp_results=nlp_results)
    session_id = "session-1"

    responses = []
    responses.append(await harness.run(session_id, "I want to book a room"))
    responses.append(await harness.run(session_id, "Dhaka"))
    responses.append(await harness.run(session_id, future_date))
    responses.append(await harness.run(session_id, "2"))
    responses.append(await harness.run(session_id, "John Doe"))

    assert responses[0]["metadata"]["action_type"] == ActionType.START_FORM.value
    assert "check-in" in responses[1]["text"].lower()
    assert "guests" in responses[2]["text"].lower()
    assert "full name" in responses[3]["text"]
    assert responses[4]["metadata"]["action_type"] == ActionType.EXECUTE_FORM.value
    assert "Booking confirmed" in responses[4]["text"]


@pytest.mark.asyncio
async def test_context_augmentation_in_pipeline():
    harness = PipelineHarness(
        enable_forms=False,
        enable_context=True,
        nlp_results=[make_faq_result("Context answer")],
        context_prefix="prior",
    )
    result = await harness.run("session-ctx", "status")

    assert harness.context_augmenter.calls[-1] == "status"
    assert result["metadata"]["context_augmentation"]["augmented"] is True


@pytest.mark.asyncio
async def test_ner_extraction_metadata():
    harness = PipelineHarness(
        enable_forms=False,
        enable_ner=True,
        enable_context=False,
        nlp_results=[make_faq_result("Context answer")],
        ner_entities={"nid_number": "12345678901234567"},
    )

    result = await harness.run("session-ner", "My nid is 12345678901234567")

    assert result["metadata"]["ner"]["entities_count"] == 1
    assert harness.ner_extractor.calls != []


@pytest.mark.asyncio
async def test_hooks_execution_order():
    harness = PipelineHarness(enable_forms=False)
    order = []

    async def before_nlp(context):
        order.append("before_nlp")

    async def after_response(context):
        order.append("after_response")

    harness.hook_manager.register(HookPoint.BEFORE_NLP, before_nlp, name="before_nlp", priority=10)
    harness.hook_manager.register(HookPoint.AFTER_RESPONSE, after_response, name="after_response", priority=10)

    await harness.run("session-hooks", "question")

    assert order == ["before_nlp", "after_response"]


@pytest.mark.asyncio
async def test_session_persistence_records_history():
    harness = PipelineHarness(enable_forms=False, nlp_results=[make_faq_result("Persisted")])
    await harness.run("session-persist", "hello")

    stored = await harness.session_manager.session_store.get("session-persist")
    assert stored is not None
    assert stored.conversation_history[-1].bot_response == "Persisted"


@pytest.mark.asyncio
async def test_policy_fallback_on_failed_nlp():
    failing_result = NLPResult(
        success=False,
        classification=[],
        search_results=[],
        strategy="error",
        confidence=0.0,
        processing_time_ms=0,
        error="model failure",
    )
    harness = PipelineHarness(enable_forms=False, nlp_results=[failing_result])
    result = await harness.run("session-fail", "?")

    assert result["success"] is True
    assert result["metadata"]["action_type"] == ActionType.FALLBACK.value
    assert "প্রক্রিয়াকরণ" in result["text"]


@pytest.mark.asyncio
async def test_error_hook_triggered_on_exception():
    failing_policy = FakePolicyEngine(scripted_actions=[])
    harness = PipelineHarness(enable_forms=False, policy_engine=failing_policy)

    errors = []

    async def on_error(ctx):
        errors.append(ctx.error)

    harness.hook_manager.register(HookPoint.ON_ERROR, on_error, name="on_error")

    # Force failure by raising inside policy engine
    async def crashing_decide(**kwargs):
        raise RuntimeError("policy crash")

    failing_policy.decide_next_action = crashing_decide

    result = await harness.run("session-error", "question")

    assert result["success"] is False
    assert any(isinstance(err, RuntimeError) for err in errors)
