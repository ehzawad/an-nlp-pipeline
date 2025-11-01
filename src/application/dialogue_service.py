"""
Enhanced Dialogue Service - Complete pipeline with all features integrated.

Integrates: NER, Context Augmentation, Hooks, Forms, Policy Engine, Session Store
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
import logging

from src.domain.events import DomainEvent, EventType, turn_started, nlp_completed
from src.shared.tenant_context import TenantContext
from src.infrastructure.tracing import trace_async, Tracer
from src.application.nlp_service import NLPResult
from src.application.hooks import HookContext, HookPoint
from src.application.session import SessionState

logger = logging.getLogger(__name__)


@dataclass
class TurnContext:
    """Context object passed through dialogue pipeline stages."""

    # Input
    session_id: str
    query: str
    correlation_id: str
    user_id: str = "anonymous"
    tenant_id: str = "default"

    # Tenant context
    tenant_context: Optional[TenantContext] = None

    # Processing state
    processed_query: str = ""
    entities: Optional[Dict[str, Any]] = None
    nlp_result: Optional[NLPResult] = None
    policy_action: Optional[Any] = None
    response_text: Optional[str] = None
    response_metadata: Dict[str, Any] = field(default_factory=dict)

    # Session state
    session_state: Optional[SessionState] = None

    # Events generated during processing
    events: List[DomainEvent] = field(default_factory=list)

    # Error handling
    error: Optional[Exception] = None

    def add_event(self, event: DomainEvent) -> None:
        """Add event to be persisted"""
        self.events.append(event)


class DialogueStage(ABC):
    """Single-responsibility stage in dialogue pipeline."""
    
    def __init__(self):
        self._next_stage: Optional[DialogueStage] = None
    
    def set_next(self, stage: "DialogueStage") -> "DialogueStage":
        """Chain stages together"""
        self._next_stage = stage
        return stage
    
    @abstractmethod
    async def process(self, context: TurnContext) -> TurnContext:
        """Process context and optionally call next stage"""
        pass
    
    async def _process_next(self, context: TurnContext) -> TurnContext:
        """Helper to call next stage"""
        if self._next_stage:
            return await self._next_stage.process(context)
        return context


class SessionLoadStage(DialogueStage):
    """Load session state from store."""
    
    @trace_async("dialogue.stage.session_load")
    async def process(self, context: TurnContext) -> TurnContext:
        """Load or create session"""
        logger.info(f"[PIPELINE] ========== STARTING PIPELINE ==========")
        logger.info(f"[PIPELINE] Session: {context.session_id} | Query: \'{context.query}\'")
        logger.info(f"[PIPELINE] User: {context.user_id} | Tenant: {context.tenant_id}")
        session_manager = await context.tenant_context.get_session_manager()
        
        session_state = await session_manager.get_or_create(
            session_id=context.session_id,
            user_id=context.user_id
        )

        context.session_state = session_state
        logger.info(f"[PIPELINE] ========== STARTING PIPELINE ==========")
        logger.info(f"[SESSION] Loaded session: {context.session_id}")
        logger.debug(f"Session loaded: {context.session_id}")
        
        return await self._process_next(context)


class EventPublishingStage(DialogueStage):
    """Publish turn started event."""
    
    @trace_async("dialogue.stage.event_publishing")
    async def process(self, context: TurnContext) -> TurnContext:
        """Publish turn started event"""
        event = turn_started(
            session_id=context.session_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            query=context.query,
            correlation_id=context.correlation_id,
        )
        
        context.add_event(event)
        
        event_bus = await context.tenant_context.get_event_bus()
        await event_bus.publish(event)
        
        return await self._process_next(context)


class HookStage(DialogueStage):
    """Execute hooks at specific point."""
    
    def __init__(self, hook_point: HookPoint):
        super().__init__()
        self.hook_point = hook_point
    
    @trace_async("dialogue.stage.hooks")
    async def process(self, context: TurnContext) -> TurnContext:
        """Execute hooks"""
        hook_manager = await context.tenant_context.get_hook_manager()
        
        # Create hook context
        hook_ctx = HookContext(
            query=context.query,
            session_id=context.session_id,
            nlp_result=context.nlp_result.__dict__ if context.nlp_result else None,
            policy_decision=context.policy_action.__dict__ if context.policy_action else None,
            response={"text": context.response_text, "metadata": context.response_metadata},
            session_state=context.session_state.to_dict() if context.session_state else None,
            error=context.error
        )
        
        # Execute hooks
        await hook_manager.execute(self.hook_point, hook_ctx)
        
        return await self._process_next(context)


class PreprocessingStage(DialogueStage):
    """Preprocess query with optional summarization and context augmentation."""
    
    def __init__(self, enable_context_augmentation: bool = True, enable_summarization: bool = True):
        super().__init__()
        self.enable_context_augmentation = enable_context_augmentation
        self.enable_summarization = enable_summarization
    
    @trace_async("dialogue.stage.preprocessing")
    async def process(self, context: TurnContext) -> TurnContext:
        """Preprocess query"""
        processed = context.query
        
        # Summarization
        if self.enable_summarization and len(context.query) > 200:
            summarizer = await context.tenant_context.get_summarizer()
            processed, meta = await summarizer.summarize(processed)
            context.response_metadata["summarization"] = meta
        
        # Context augmentation
        if self.enable_context_augmentation and context.session_state:
            context_augmenter = await context.tenant_context.get_context_augmenter()
            processed, meta = await context_augmenter.augment_if_needed(
                processed, context.session_state
            )
            if meta.get("augmented"):
                context.response_metadata["context_augmentation"] = meta
        
        # Normalization
        processed = processed.strip()
        context.processed_query = processed
        
        logger.debug(f"Preprocessed query: '{context.query}' -> '{processed}'")
        
        return await self._process_next(context)


class NERExtractionStage(DialogueStage):
    """Extract named entities from query."""
    
    def __init__(self, enable_ner: bool = True):
        super().__init__()
        self.enable_ner = enable_ner
    
    @trace_async("dialogue.stage.ner")
    async def process(self, context: TurnContext) -> TurnContext:
        """Extract entities"""
        if not self.enable_ner:
            return await self._process_next(context)
        
        ner_extractor = await context.tenant_context.get_ner_extractor()
        
        # Get form slots if in form
        form_slots = None
        if context.session_state and context.session_state.active_form:
            form_registry = await context.tenant_context.get_form_registry()
            form = form_registry.get(context.session_state.active_form.form_name)
            if form:
                form_slots = [slot.name for slot in form.required_slots()]
        
        # Extract entities
        result = await ner_extractor.extract(context.processed_query or context.query, form_slots)
        context.entities = result["entities"]
        context.response_metadata["ner"] = {
            "method": result["method"],
            "entities_count": len(result["entities"])
        }
        
        logger.debug(f"Extracted {len(context.entities)} entities")
        
        return await self._process_next(context)


class NLPStage(DialogueStage):
    """Run NLP pipeline."""
    
    @trace_async("dialogue.stage.nlp")
    async def process(self, context: TurnContext) -> TurnContext:
        """Run NLP pipeline"""
        logger.info(f"[PIPELINE] Stage: NLP Processing")
        logger.debug(f"[FLOW] → NLPStage: Classification + Semantic Search")
        nlp_service = await context.tenant_context.get_nlp_service()
        
        query = context.processed_query or context.query
        
        nlp_result = await nlp_service.process(
            query=query,
            tenant_id=context.tenant_id,
            top_k=5
        )
        
        context.nlp_result = nlp_result
        
        # Create NLP completed event
        event = nlp_completed(
            session_id=context.session_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            nlp_result={
                "confidence": nlp_result.confidence,
                "strategy": nlp_result.strategy,
                "top_cluster": nlp_result.classification[0].cluster if nlp_result.classification else None,
            },
            correlation_id=context.correlation_id,
            causation_id=context.events[0].event_id if context.events else None,
        )
        
        context.add_event(event)
        
        event_bus = await context.tenant_context.get_event_bus()
        await event_bus.publish(event)
        
        return await self._process_next(context)


class PolicyStage(DialogueStage):
    """Policy decision."""
    
    @trace_async("dialogue.stage.policy")
    async def process(self, context: TurnContext) -> TurnContext:
        """Make policy decision"""
        if not context.nlp_result or not context.nlp_result.success:
            # NLP failed, use fallback
            from src.application.policy import Action
            context.policy_action = Action.fallback()
            return await self._process_next(context)
        
        policy_engine = await context.tenant_context.get_policy_engine()
        
        # Convert NLPResult to dict for policy engine
        nlp_result_dict = {
            "confidence": context.nlp_result.confidence,
            "classification": [{"cluster": c.cluster, "confidence": c.confidence} 
                             for c in context.nlp_result.classification],
            "search_results": [{"question": r.question, "cluster": r.cluster, 
                              "tag": r.tag, "score": r.score} 
                             for r in context.nlp_result.search_results],
            "strategy": context.nlp_result.strategy
        }
        
        action = await policy_engine.decide_next_action(
            session_state=context.session_state,
            user_query=context.query,
            nlp_result=nlp_result_dict,
            entities=context.entities
        )
        
        context.policy_action = action
        logger.info(f"Policy decision: {action.action_type.value}")
        
        return await self._process_next(context)


class ResponseGenerationStage(DialogueStage):
    """Generate response based on policy decision."""
    
    @trace_async("dialogue.stage.response_generation")
    async def process(self, context: TurnContext) -> TurnContext:
        """Generate user-facing response"""
        if not context.policy_action:
            context.response_text = "দুঃখিত, একটি সমস্যা হয়েছে।"
            return await self._process_next(context)
        
        from src.application.policy import ActionType
        action = context.policy_action
        
        # Use pre-generated response if available
        if action.response_text:
            context.response_text = action.response_text
        else:
            # Format based on action type
            if action.action_type == ActionType.FAQ_ANSWER:
                results = action.data.get("results", [])
                if results:
                    context.response_text = results[0].get("question", "উত্তর পাওয়া যায়নি")
            elif action.action_type == ActionType.CLARIFY_INTENT:
                context.response_text = "আমি নিশ্চিত নই। আপনি কি জানতে চান?"
            elif action.action_type == ActionType.ESCALATE:
                context.response_text = "দুঃখিত, আমি এই বিষয়ে সাহায্য করতে পারছি না। একজন এজেন্টের সাথে যোগাযোগ করুন।"
                logger.warning(f"Escalation triggered: {action.data}")
            elif action.action_type == ActionType.FALLBACK:
                # FALLBACK gets generic processing message (for tests compatibility)
                context.response_text = "প্রক্রিয়াকরণ..."
                logger.warning(f"Fallback triggered: {action.data}")
            else:
                context.response_text = "প্রক্রিয়াকরণ..."
                logger.warning(f"Unhandled action type: {action.action_type.value}")
        
        context.response_metadata["action_type"] = action.action_type.value
        
        return await self._process_next(context)


class SessionPersistenceStage(DialogueStage):
    """Persist events and session state."""
    
    @trace_async("dialogue.stage.persistence")
    async def process(self, context: TurnContext) -> TurnContext:
        """Persist events and session"""
        # Persist events to event store
        event_store = await context.tenant_context.get_event_store()
        for event in context.events:
            await event_store.append(event)
        
        # Update session state
        if context.session_state:
            context.session_state.add_turn(
                user_query=context.query,
                bot_response=context.response_text or "",
                nlp_result=asdict(context.nlp_result) if context.nlp_result else None,
                action_taken=context.policy_action.action_type.value if context.policy_action else None,
                entities_extracted=context.entities
            )
            
            # Save session
            session_manager = await context.tenant_context.get_session_manager()
            await session_manager.update(context.session_state)
        
        logger.debug(f"Persisted {len(context.events)} events and updated session")
        
        return await self._process_next(context)


class EnhancedDialoguePipeline:
    """
    Enhanced dialogue pipeline with all features.
    
    Stages:
    1. Session Load
    2. Event Publishing
    3. Hook: BEFORE_NLP
    4. Preprocessing (summarization, context augmentation)
    5. NER Extraction
    6. NLP
    7. Hook: AFTER_NLP
    8. Hook: BEFORE_POLICY
    9. Policy
    10. Hook: AFTER_POLICY
    11. Response Generation
    12. Hook: BEFORE_RESPONSE
    13. Session Persistence
    14. Hook: AFTER_RESPONSE
    """
    
    def __init__(
        self,
        enable_forms: bool = True,
        enable_ner: bool = True,
        enable_context_augmentation: bool = True,
        enable_summarization: bool = True,
    ):
        self.pipeline = self._build_pipeline(
            enable_forms,
            enable_ner,
            enable_context_augmentation,
            enable_summarization
        )
    
    def _build_pipeline(
        self,
        enable_forms: bool,
        enable_ner: bool,
        enable_context_augmentation: bool,
        enable_summarization: bool,
    ) -> DialogueStage:
        """Build complete pipeline."""
        
        # Stage 1: Session load
        stage1 = SessionLoadStage()
        
        # Stage 2: Event publishing
        stage2 = EventPublishingStage()
        stage1.set_next(stage2)
        
        # Stage 3: Hook BEFORE_NLP
        stage3 = HookStage(HookPoint.BEFORE_NLP)
        stage2.set_next(stage3)
        
        # Stage 4: Preprocessing
        stage4 = PreprocessingStage(enable_context_augmentation, enable_summarization)
        stage3.set_next(stage4)
        
        # Stage 5: NER
        stage5 = NERExtractionStage(enable_ner)
        stage4.set_next(stage5)
        
        # Stage 6: NLP
        stage6 = NLPStage()
        stage5.set_next(stage6)
        
        # Stage 7: Hook AFTER_NLP
        stage7 = HookStage(HookPoint.AFTER_NLP)
        stage6.set_next(stage7)
        
        # Stage 8: Hook BEFORE_POLICY
        stage8 = HookStage(HookPoint.BEFORE_POLICY)
        stage7.set_next(stage8)
        
        # Stage 9: Policy
        stage9 = PolicyStage()
        stage8.set_next(stage9)
        
        # Stage 10: Hook AFTER_POLICY
        stage10 = HookStage(HookPoint.AFTER_POLICY)
        stage9.set_next(stage10)
        
        # Stage 11: Response generation
        stage11 = ResponseGenerationStage()
        stage10.set_next(stage11)
        
        # Stage 12: Hook BEFORE_RESPONSE
        stage12 = HookStage(HookPoint.BEFORE_RESPONSE)
        stage11.set_next(stage12)
        
        # Stage 13: Persistence
        stage13 = SessionPersistenceStage()
        stage12.set_next(stage13)
        
        # Stage 14: Hook AFTER_RESPONSE
        stage14 = HookStage(HookPoint.AFTER_RESPONSE)
        stage13.set_next(stage14)
        
        return stage1  # Return first stage
    
    @trace_async("dialogue.process_turn")
    async def process_turn(
        self,
        query: str,
        session_id: str,
        tenant_context: TenantContext,
        correlation_id: str,
        user_id: str = "anonymous",
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        """Process dialogue turn through complete pipeline."""
        # Create context
        context = TurnContext(
            session_id=session_id,
            query=query,
            correlation_id=correlation_id,
            tenant_context=tenant_context,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        
        try:
            # Run through pipeline
            result_context = await self.pipeline.process(context)
            
            return {
                "text": result_context.response_text,
                "metadata": result_context.response_metadata,
                "success": True,
            }
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            
            # Execute error hooks
            hook_manager = await tenant_context.get_hook_manager()
            hook_ctx = HookContext(
                query=query,
                session_id=session_id,
                error=e
            )
            await hook_manager.execute(HookPoint.ON_ERROR, hook_ctx)
            
            return {
                "text": "দুঃখিত, একটি সমস্যা হয়েছে।",
                "metadata": {"error": str(e)},
                "success": False,
            }

