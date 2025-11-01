"""Policy handlers using Chain of Responsibility pattern."""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import logging
from .base_policy import Action, ActionType

logger = logging.getLogger(__name__)


class PolicyHandler(ABC):
    """Base handler in policy chain."""

    def __init__(self):
        self._next_handler: Optional[PolicyHandler] = None

    def set_next(self, handler: "PolicyHandler") -> "PolicyHandler":
        """Set next handler in chain."""
        self._next_handler = handler
        return handler

    async def handle(
        self,
        session_state,
        user_query: str,
        nlp_result: dict,
        entities: Optional[Dict[str, Any]] = None
    ) -> Optional[Action]:
        """Handle request or pass to next handler."""
        logger.debug(f"[HANDLER] {self.__class__.__name__} processing...")
        action = await self._process(session_state, user_query, nlp_result, entities)

        if action:
            return action

        if self._next_handler:
            return await self._next_handler.handle(
                session_state, user_query, nlp_result, entities
            )

        return None

    @abstractmethod
    async def _process(
        self,
        session_state,
        user_query: str,
        nlp_result: dict,
        entities: Optional[Dict[str, Any]] = None
    ) -> Optional[Action]:
        """Process request. Return Action if handled, None to pass to next."""
        pass


class ActiveFormHandler(PolicyHandler):
    """Handle active form slot collection."""

    def __init__(self, form_policy, form_registry=None):
        super().__init__()
        self.form_policy = form_policy
        self.form_registry = form_registry

    async def _process(self, session_state, user_query, nlp_result, entities) -> Optional[Action]:
        """If form is active, handle slot collection with interruption detection."""
        has_active_form = session_state.active_form is not None
        logger.info(f"[HANDLER] ActiveFormHandler: active_form={has_active_form}")
        
        if not has_active_form:
            logger.debug("ActiveFormHandler: No active form, passing to next handler")
            return None
            
        logger.info(f"ActiveFormHandler: Form '{session_state.active_form.form_name}' is active")
        
        # Check for interruption before form processing
        if self.form_registry:
            from src.application.forms.interruption_handler import FormInterruptionHandler
            
            form = self.form_registry.get(session_state.active_form.form_name)
            if form and hasattr(form, 'trigger_tags'):
                form_tags = form.trigger_tags
                nlp_tags = [r.get('tag') for r in nlp_result.get('search_results', []) if r.get('tag')]
                
                if form_tags and nlp_tags:
                    interruption_handler = FormInterruptionHandler()
                    is_interruption = interruption_handler.detect_interruption(form_tags, nlp_tags)
                    
                    if is_interruption:
                        logger.info(f"Interruption detected! Form tags: {form_tags}, NLP tags: {nlp_tags}")
                        # For now, log but continue with form - full interruption flow can be added later
                        # return Action(
                        #     action_type=ActionType.ASK_INTERRUPTION_CHOICE,
                        #     response_text=interruption_handler.create_choice_prompt(
                        #         form.form_name, session_state.active_form.current_slot
                        #     )
                        # )
        
        # Continue with form slot collection
        return await self.form_policy.decide(session_state, user_query, nlp_result, entities)


class FormTriggerHandler(PolicyHandler):
    """Detect and trigger forms based on tags."""

    def __init__(self, form_registry):
        super().__init__()
        self.form_registry = form_registry

    async def _process(self, session_state, user_query, nlp_result, entities) -> Optional[Action]:
        """Check if NLP result triggers a form."""
        results = nlp_result.get("search_results") or nlp_result.get("results") or []
        
        logger.debug(f"[HANDLER] FormTriggerHandler: {len(results)} search results")
        
        # Check if any result has a tag that triggers a form
        for result in results:
            tag = result.get("tag")
            logger.debug(f"[HANDLER] FormTriggerHandler: checking tag='{tag}'")
            if tag:
                form = self.form_registry.get_by_tag(tag)
                if form:
                    logger.info(f"FormTriggerHandler: Tag '{tag}' triggers form '{form.form_name}'")

                    # Start form
                    from ..forms import FormRunner, FormState, FormStatus
                    form_runner = FormRunner()
                    form_state = await form_runner.start_form(form)

                    # Save to session
                    from ..session.models import FormStateData
                    session_state.active_form = FormStateData(
                        form_name=form.form_name,
                        current_slot=form_state.current_slot,
                        filled_slots=form_state.filled_slots
                    )

                    # Get the prompt for the first slot
                    first_slot_obj = form.get_slot_by_name(form_state.current_slot)
                    prompt = first_slot_obj.prompt if first_slot_obj else "Please provide information."
                    
                    # Combine welcome message with first slot prompt
                    welcome_msg = form.on_form_start()
                    full_response = f"{welcome_msg}\n\n{prompt}" if welcome_msg else prompt

                    return Action(
                        action_type=ActionType.START_FORM,
                        data={
                            "form_name": form.form_name,
                            "first_slot": form_state.current_slot
                        },
                        response_text=full_response,
                        metadata={"form_started": True, "slot_requested": form_state.current_slot}
                    )

        return None


class HighConfidenceFAQHandler(PolicyHandler):
    """Handle high-confidence FAQ queries."""

    def __init__(self, faq_policy):
        super().__init__()
        self.faq_policy = faq_policy

    async def _process(self, session_state, user_query, nlp_result, entities) -> Optional[Action]:
        """If high confidence, return FAQ answer."""
        confidence = self.faq_policy._get_confidence(nlp_result)

        logger.info(f"[HANDLER] HighConfidenceFAQHandler: confidence={confidence:.3f} >= threshold={self.faq_policy.confidence_threshold:.3f}")
        if confidence >= self.faq_policy.confidence_threshold:
            logger.info(f"HighConfidenceFAQHandler: High confidence ({confidence:.3f})")
            return await self.faq_policy.decide(session_state, user_query, nlp_result, entities)

        return None


class ClarificationFallbackHandler(PolicyHandler):
    """Fallback to clarification or escalation."""

    def __init__(self, clarification_policy):
        super().__init__()
        self.clarification_policy = clarification_policy

    async def _process(self, session_state, user_query, nlp_result, entities) -> Optional[Action]:
        """Always returns an action (last resort)."""
        logger.info("ClarificationFallbackHandler: Handling as final fallback")
        return await self.clarification_policy.decide(
            session_state, user_query, nlp_result, entities
        )


class EscalationHandler(PolicyHandler):
    """Handle escalation scenarios."""

    async def _process(self, session_state, user_query, nlp_result, entities) -> Optional[Action]:
        """Check if we should escalate."""
        if session_state.escalation_flag:
            logger.warning("EscalationHandler: Escalation flag set")
            return Action.escalate(reason="Max errors exceeded")

        return None

