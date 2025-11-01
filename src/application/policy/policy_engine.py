"""Policy engine for dialogue flow control."""

from typing import Optional, Dict, Any
import logging
from .base_policy import Action
from .faq_policy import FAQPolicy
from .clarification_policy import ClarificationPolicy
from .form_policy import FormPolicy
from .handlers import (
    EscalationHandler,
    ActiveFormHandler,
    FormTriggerHandler,
    HighConfidenceFAQHandler,
    ClarificationFallbackHandler,
)

logger = logging.getLogger(__name__)


class PolicyEngine:
    """
    Orchestrate policies to decide dialogue flow.
    
    Routes to appropriate policy based on:
    - Session state (active task, forms, etc.)
    - NLP confidence
    - User intent
    """

    def __init__(
        self,
        handler_chain = None,
        confidence_threshold: float = 0.7,
        enable_forms: bool = True,
        form_registry = None
    ):
        """
        Initialize policy engine.

        Args:
            handler_chain: Pre-built handler chain (optional). If provided, other params are ignored.
            confidence_threshold: Threshold for FAQ confidence
            enable_forms: Whether to enable form handling
            form_registry: Form registry (required if enable_forms=True)
        """
        # If a pre-built handler chain is provided, use it directly
        if handler_chain is not None:
            self._handler_chain = handler_chain
            logger.info("PolicyEngine initialized with pre-built handler chain")
            return

        # Otherwise, build the chain from parameters
        self.confidence_threshold = confidence_threshold
        self.enable_forms = enable_forms

        # Initialize core policies
        self.faq_policy = FAQPolicy(confidence_threshold=confidence_threshold)
        self.clarification_policy = ClarificationPolicy()

        # Initialize form policy if enabled
        if enable_forms and form_registry:
            from ..forms import FormRunner
            self.form_registry = form_registry
            self.form_runner = FormRunner()
            self.form_policy = FormPolicy(
                form_runner=self.form_runner,
                form_registry=self.form_registry
            )
            logger.info(f"Forms feature enabled ({len(form_registry)} form(s) registered)")
        else:
            self.form_runner = None
            self.form_registry = None
            self.form_policy = None
            logger.info("Forms feature disabled")

        # Build handler chain
        self._handler_chain = self._build_handler_chain()

    def _build_handler_chain(self):
        """Build chain of responsibility for policy decisions."""
        # Priority order:
        # 1. Escalation (if needed)
        # 2. Active Form (if in form)
        # 3. Form Trigger (if tag detected)
        # 4. High Confidence FAQ
        # 5. Clarification/Fallback

        escalation = EscalationHandler()

        if self.enable_forms and self.form_policy:
            active_form = ActiveFormHandler(self.form_policy, self.form_registry)
            form_trigger = FormTriggerHandler(self.form_registry)
            high_conf_faq = HighConfidenceFAQHandler(self.faq_policy)
            clarification = ClarificationFallbackHandler(self.clarification_policy)

            # Chain them
            escalation.set_next(active_form)
            active_form.set_next(form_trigger)
            form_trigger.set_next(high_conf_faq)
            high_conf_faq.set_next(clarification)
        else:
            # Without forms: escalation → FAQ → clarification
            high_conf_faq = HighConfidenceFAQHandler(self.faq_policy)
            clarification = ClarificationFallbackHandler(self.clarification_policy)

            escalation.set_next(high_conf_faq)
            high_conf_faq.set_next(clarification)

        return escalation

    async def decide_next_action(
        self,
        session_state,
        user_query: str,
        nlp_result: Optional[dict],
        entities: Optional[Dict[str, Any]] = None
    ) -> Action:
        """
        Decide what action to take next.
        
        Args:
            session_state: Current session state
            user_query: User's input
            nlp_result: NLP pipeline results
            entities: Extracted entities (optional)
            
        Returns:
            Action to execute
        """
        if not nlp_result:
            logger.warning("No NLP result provided, using fallback")
            return Action.fallback()

        logger.debug(
            f"PolicyEngine: Deciding action for query='{user_query[:50]}...', "
            f"active_form={session_state.active_form is not None}"
        )

        # Run through handler chain
        action = await self._handler_chain.handle(
            session_state, user_query, nlp_result, entities
        )

        if not action:
            logger.error("No handler returned an action, using fallback")
            action = Action.fallback()

        logger.info(f"PolicyEngine → Action: {action.action_type.value}")
        return action

