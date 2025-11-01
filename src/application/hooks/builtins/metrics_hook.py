"""Built-in metrics hook."""

from datetime import datetime
import logging
from ..manager import HookContext, HookPoint
from ..decorators import hook

logger = logging.getLogger(__name__)


@hook(HookPoint.BEFORE_NLP, priority=20)
async def metrics_start_timer(context: HookContext):
    """Start timing NLP processing."""
    context.set('nlp_start_time', datetime.now())


@hook(HookPoint.AFTER_NLP, priority=20)
async def metrics_end_timer(context: HookContext):
    """End timing NLP processing."""
    start_time = context.get('nlp_start_time')
    if start_time:
        duration = (datetime.now() - start_time).total_seconds()
        context.set('nlp_duration', duration)
        logger.info(f"[METRICS] NLP processing took {duration:.3f}s")


@hook(HookPoint.AFTER_RESPONSE, priority=20)
async def metrics_log_turn(context: HookContext):
    """Log turn completion metrics."""
    nlp_duration = context.get('nlp_duration', 0)
    logger.info(
        f"[METRICS] Turn completed - "
        f"Session: {context.session_id}, "
        f"NLP: {nlp_duration:.3f}s"
    )

