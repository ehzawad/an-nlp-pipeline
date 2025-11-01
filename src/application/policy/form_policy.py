"""Form policy for handling form-based dialogues."""

from typing import Optional, Dict, Any
import logging
from .base_policy import BasePolicy, Action, ActionType

logger = logging.getLogger(__name__)


class FormPolicy(BasePolicy):
    """
    Policy for handling form execution.
    
    Manages:
    - Slot collection
    - Validation and retries
    - Form completion and execution
    """

    def __init__(self, form_runner, form_registry):
        """Initialize form policy."""
        self.form_runner = form_runner
        self.form_registry = form_registry

    async def decide(
        self,
        session_state,
        user_query: str,
        nlp_result: dict,
        entities: Optional[Dict[str, Any]] = None
    ) -> Action:
        """Decide next action for form execution."""
        if not session_state.active_form:
            logger.error("FormPolicy called but no active form in session")
            return Action.fallback()

        form_state = session_state.active_form
        form = self.form_registry.get(form_state.form_name)

        if not form:
            logger.error(f"Form '{form_state.form_name}' not found in registry")
            return Action.escalate(reason=f"Form definition not found: {form_state.form_name}")

        logger.debug(
            f"FormPolicy: form={form_state.form_name}, "
            f"current_slot={form_state.current_slot}"
        )

        # Collect slot value
        try:
            updated_state, response_text = await self.form_runner.collect_slot(
                form, form_state, user_query
            )

            # Update session's form state
            session_state.active_form = updated_state

            # Check if form is complete
            if updated_state.current_slot is None and updated_state.filled_slots:
                # Form complete, execute it
                logger.info(f"Form {form_state.form_name} complete, executing...")

                result_state, execution_result = await self.form_runner.execute_form(
                    form, updated_state
                )

                session_state.active_form = None  # Clear active form

                return Action(
                    action_type=ActionType.EXECUTE_FORM,
                    data={
                        "form_name": form_state.form_name,
                        "execution_result": execution_result
                    },
                    response_text=execution_result.get("message", "Form executed successfully."),
                    metadata={"form_completed": True}
                )

            # Still collecting slots
            return Action(
                action_type=ActionType.COLLECT_SLOT,
                data={
                    "form_name": form_state.form_name,
                    "slot_name": updated_state.current_slot,
                    "filled_slots": updated_state.filled_slots
                },
                response_text=response_text,
                metadata={"collecting_slots": True}
            )

        except Exception as e:
            logger.error(f"Form execution error: {e}")
            return Action.escalate(reason=f"Form error: {str(e)}")

