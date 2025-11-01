"""Built-in logging hook."""

import logging
from ..manager import HookContext, HookPoint
from ..decorators import hook

logger = logging.getLogger(__name__)


@hook(HookPoint.BEFORE_NLP, priority=10)
async def log_before_nlp(context: HookContext):
    """Log before NLP processing."""
    logger.info(f"[HOOK] Before NLP - Query: {context.query[:100]}...")


@hook(HookPoint.AFTER_NLP, priority=10)
async def log_after_nlp(context: HookContext):
    """Log after NLP processing."""
    if context.nlp_result:
        confidence = context.nlp_result.get('confidence', 0)
        logger.info(f"[HOOK] After NLP - Confidence: {confidence:.2f}")


@hook(HookPoint.BEFORE_POLICY, priority=10)
async def log_before_policy(context: HookContext):
    """Log before policy decision."""
    logger.info(f"[HOOK] Before Policy - Session: {context.session_id}")


@hook(HookPoint.AFTER_POLICY, priority=10)
async def log_after_policy(context: HookContext):
    """Log after policy decision."""
    if context.policy_decision:
        action = context.policy_decision.get('action', 'unknown')
        logger.info(f"[HOOK] After Policy - Action: {action}")


@hook(HookPoint.ON_ERROR, priority=10)
async def log_error(context: HookContext):
    """Log errors."""
    if context.error:
        logger.error(f"[HOOK] Error: {context.error}")

