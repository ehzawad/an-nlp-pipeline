"""In-memory session store using Python dict."""

import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
from .models import SessionState

logger = logging.getLogger(__name__)


class InMemorySessionStore:
    """
    In-memory session storage using Python dict.
    
    - Sessions stored in memory (lost on restart)
    - TTL-based expiration
    - Perfect for single-server deployments
    - Zero external dependencies
    """

    def __init__(
        self,
        redis_host: str = None,  # Ignored, kept for backward compatibility
        redis_port: int = None,  # Ignored, kept for backward compatibility
        redis_db: int = None,  # Ignored, kept for backward compatibility
        ttl_seconds: int = 1800,
        key_prefix: str = "dialogue:session:"  # Ignored, kept for backward compatibility
    ):
        """
        Initialize in-memory session store.
        
        Args:
            redis_host: Ignored (kept for backward compatibility)
            redis_port: Ignored (kept for backward compatibility)
            redis_db: Ignored (kept for backward compatibility)
            ttl_seconds: Session TTL in seconds (default: 30 minutes)
            key_prefix: Ignored (kept for backward compatibility)
        """
        self._sessions: Dict[str, SessionState] = {}
        self._expiry: Dict[str, datetime] = {}
        self.ttl_seconds = ttl_seconds
        logger.info(f"InMemorySessionStore initialized (TTL: {ttl_seconds}s)")

    def _is_expired(self, session_id: str) -> bool:
        """Check if session has expired."""
        if session_id not in self._expiry:
            return True
        return datetime.now() >= self._expiry[session_id]

    def _cleanup_expired(self):
        """Remove expired sessions (called periodically)."""
        now = datetime.now()
        expired = [sid for sid, exp_time in self._expiry.items() if now >= exp_time]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._expiry.pop(sid, None)
            logger.debug(f"Cleaned up expired session: {sid}")

    async def get(self, session_id: str) -> Optional[SessionState]:
        """Get session state."""
        # Cleanup expired sessions opportunistically
        self._cleanup_expired()
        
        if session_id not in self._sessions or self._is_expired(session_id):
            return None
        
        return self._sessions[session_id]

    async def set(self, session_id: str, session_state: SessionState) -> bool:
        """Save session state."""
        try:
            self._sessions[session_id] = session_state
            self._expiry[session_id] = datetime.now() + timedelta(seconds=self.ttl_seconds)
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    async def delete(self, session_id: str) -> bool:
        """Delete session."""
        try:
            self._sessions.pop(session_id, None)
            self._expiry.pop(session_id, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        if session_id not in self._sessions:
            return False
        return not self._is_expired(session_id)

    async def touch(self, session_id: str) -> bool:
        """Refresh session TTL."""
        try:
            if session_id in self._sessions:
                self._expiry[session_id] = datetime.now() + timedelta(seconds=self.ttl_seconds)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to touch session {session_id}: {e}")
            return False

    def count(self) -> int:
        """Get number of active sessions."""
        self._cleanup_expired()
        return len(self._sessions)
