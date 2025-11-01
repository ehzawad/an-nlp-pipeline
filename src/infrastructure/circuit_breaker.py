"""
Circuit Breaker Pattern - Prevent cascading failures from external services.

States:
- CLOSED: Normal operation, requests go through
- OPEN: Service is down, fail fast without calling
- HALF_OPEN: Testing if service recovered

Prevents:
- Thread pool exhaustion
- Cascading failures
- Wasted resources on failing services
"""

from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import asyncio
from functools import wraps


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Service down, failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN"""
    pass


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring"""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None

    state_changed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls


class CircuitBreaker:
    """
    Circuit breaker for external service calls.
    
    Monitors failures and automatically opens circuit when
    failure threshold is reached, preventing cascading failures.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        success_threshold: int = 2,  # Successes needed to close from half-open
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name (for logging)
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds to wait before trying again (OPEN → HALF_OPEN)
            expected_exception: Exception type to catch
            success_threshold: Successes needed to close from half-open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        return self._stats
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Raises:
            CircuitBreakerError: If circuit is OPEN
        """
        async with self._lock:
            # Check if we should transition from OPEN → HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable."
                    )
            
            # Try the call
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success!
                await self._on_success()
                return result
            
            except self.expected_exception as e:
                # Expected failure
                await self._on_failure()
                raise
    
    async def _on_success(self) -> None:
        """Handle successful call"""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        self._stats.last_success_time = datetime.now(timezone.utc)
        
        # If in HALF_OPEN, check if we can close
        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.success_threshold:
                self._transition_to_closed()
    
    async def _on_failure(self) -> None:
        """Handle failed call"""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._stats.last_failure_time = datetime.now(timezone.utc)
        
        # Check if we should open circuit
        if self._stats.consecutive_failures >= self.failure_threshold:
            self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try recovery"""
        if self._stats.last_failure_time is None:
            return True

        time_since_failure = datetime.now(timezone.utc) - self._stats.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _transition_to_open(self) -> None:
        """Transition to OPEN state"""
        self._state = CircuitState.OPEN
        self._stats.state_changed_at = datetime.now(timezone.utc)
        print(f"⚠️ Circuit breaker '{self.name}' opened after "
              f"{self._stats.consecutive_failures} consecutive failures")
    
    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state"""
        self._state = CircuitState.HALF_OPEN
        self._stats.state_changed_at = datetime.now(timezone.utc)
        self._stats.consecutive_successes = 0
        print(f"🔄 Circuit breaker '{self.name}' transitioned to HALF_OPEN "
              f"(testing recovery)")
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state"""
        self._state = CircuitState.CLOSED
        self._stats.state_changed_at = datetime.now(timezone.utc)
        self._stats.consecutive_failures = 0
        print(f"✅ Circuit breaker '{self.name}' closed (service recovered)")


# ============================================================================
# Decorator for Circuit Breaker
# ============================================================================

def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
):
    """
    Decorator to wrap function with circuit breaker.
    
    Usage:
        @with_circuit_breaker("nid_api", failure_threshold=5, recovery_timeout=60)
        async def call_nid_api(data: dict):
            response = await httpx.post("https://nid-api.gov.bd", json=data)
            return response.json()
    
    If NID API goes down, circuit opens and fails fast instead of hanging.
    """
    # Create circuit breaker instance (shared across calls)
    breaker = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# ============================================================================
# Retry with Exponential Backoff
# ============================================================================

async def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
) -> Any:
    """
    Retry function with exponential backoff.
    
    Often used together with circuit breaker:
    1. Circuit breaker prevents cascading failures
    2. Retry handles transient failures
    
    Usage:
        result = await retry_with_backoff(
            lambda: call_external_api(),
            max_attempts=3
        )
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        
        except Exception as e:
            last_exception = e
            
            if attempt < max_attempts - 1:
                # Wait with exponential backoff
                await asyncio.sleep(min(delay, max_delay))
                delay *= backoff_factor
    
    # All attempts failed
    raise last_exception
