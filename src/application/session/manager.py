"""Session manager for CRUD operations."""

from typing import Optional
from datetime import datetime
import logging
from .models import SessionState
from .store import InMemorySessionStore

# Type alias for backward compatibility (InMemorySessionStore is the only implementation now)
SessionStore = InMemorySessionStore

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage session CRUD operations."""

    def __init__(self, session_store: SessionStore):
        """Initialize session manager."""
        self.session_store = session_store

    async def get_or_create(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> SessionState:
        """Get existing session or create new one."""
        # Try to get existing
        session = await self.session_store.get(session_id)
        
        if session is not None:
            # Refresh TTL
            await self.session_store.touch(session_id)
            has_active_form = session.active_form is not None
            logger.info(f"[SESSION] Loaded existing session: {session_id} - active_form={has_active_form}")
            if has_active_form:
                logger.info(f"[SESSION] Loaded form: {session.active_form.form_name}, slot={session.active_form.current_slot}")
            return session
        
        # Create new session
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        await self.session_store.set(session_id, session)
        logger.info(f"Created new session: {session_id}")
        
        return session

    async def update(self, session_state: SessionState) -> bool:
        """Update session state."""
        session_state.last_activity = datetime.now()
        
        # Log session state for debugging
        has_active_form = session_state.active_form is not None
        logger.info(f"[SESSION] Saving session {session_state.session_id} - active_form={has_active_form}")
        if has_active_form:
            logger.info(f"[SESSION] Form details: {session_state.active_form.form_name}, slot={session_state.active_form.current_slot}")
        
        success = await self.session_store.set(session_state.session_id, session_state)
        
        if success:
            logger.info(f"[SESSION] Successfully saved session: {session_state.session_id}")
        else:
            logger.error(f"[SESSION] Failed to save session: {session_state.session_id}")
        
        return success

    async def delete(self, session_id: str) -> bool:
        """Delete session."""
        success = await self.session_store.delete(session_id)
        
        if success:
            logger.info(f"Deleted session: {session_id}")
        else:
            logger.error(f"Failed to delete session: {session_id}")
        
        return success

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return await self.session_store.exists(session_id)

