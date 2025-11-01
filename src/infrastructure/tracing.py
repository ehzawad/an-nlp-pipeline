"""
Distributed Tracing with OpenTelemetry.

Provides:
- Request tracing across all components
- Parent-child span relationships
- Performance monitoring
- Error tracking
- Distributed context propagation
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextvars import ContextVar
import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum


# Context variable for trace context (thread-safe)
_trace_context: ContextVar[Optional["TraceContext"]] = ContextVar(
    "trace_context",
    default=None
)


class SpanStatus(str, Enum):
    """Span status"""
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """
    Represents a unit of work in distributed trace.
    
    Simplified OpenTelemetry span for demo.
    Production would use actual OpenTelemetry SDK.
    """
    
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    
    start_time: float
    end_time: Optional[float] = None
    
    status: SpanStatus = SpanStatus.OK
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: list[Dict[str, Any]] = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute (metadata)"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        """Add event to span (log point)"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })
    
    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set span status"""
        self.status = status
        if description:
            self.attributes["status_description"] = description
    
    def end(self) -> None:
        """End span and record duration"""
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0


@dataclass
class TraceContext:
    """
    Trace context for distributed tracing.
    
    Propagated across all async calls and service boundaries.
    """
    
    trace_id: str
    parent_span_id: Optional[str] = None
    
    # Stack of active spans
    _span_stack: list[Span] = field(default_factory=list)
    
    def start_span(self, name: str) -> Span:
        """Start a new span"""
        import secrets
        
        span = Span(
            name=name,
            trace_id=self.trace_id,
            span_id=secrets.token_urlsafe(16),
            parent_span_id=self.parent_span_id,
            start_time=time.time(),
        )
        
        self._span_stack.append(span)
        self.parent_span_id = span.span_id
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End span and pop from stack"""
        span.end()
        
        if self._span_stack and self._span_stack[-1] == span:
            self._span_stack.pop()
            
            # Restore parent span as current
            if self._span_stack:
                self.parent_span_id = self._span_stack[-1].span_id
            else:
                self.parent_span_id = None
    
    @property
    def current_span(self) -> Optional[Span]:
        """Get current active span"""
        if self._span_stack:
            return self._span_stack[-1]
        return None


class Tracer:
    """
    Distributed tracer.
    
    Manages trace context and span lifecycle.
    """
    
    @staticmethod
    def get_current_context() -> Optional[TraceContext]:
        """Get current trace context from context variable"""
        return _trace_context.get()
    
    @staticmethod
    def set_context(context: TraceContext) -> None:
        """Set trace context"""
        _trace_context.set(context)
    
    @staticmethod
    def start_trace(trace_id: str) -> TraceContext:
        """Start a new trace"""
        context = TraceContext(trace_id=trace_id)
        _trace_context.set(context)
        return context
    
    @staticmethod
    def start_span(name: str) -> Optional[Span]:
        """Start a span in current trace"""
        context = _trace_context.get()
        if context:
            return context.start_span(name)
        return None
    
    @staticmethod
    def end_span(span: Span) -> None:
        """End a span"""
        context = _trace_context.get()
        if context:
            context.end_span(span)
    
    @staticmethod
    def current_span() -> Optional[Span]:
        """Get current active span"""
        context = _trace_context.get()
        if context:
            return context.current_span
        return None


# ============================================================================
# Decorator for Automatic Tracing
# ============================================================================

def trace_async(span_name: Optional[str] = None):
    """
    Decorator to automatically trace async functions.
    
    Usage:
        @trace_async("nlp.classify")
        async def classify(query: str):
            ...
    
    Creates span automatically, records timing, and captures errors.
    """
    def decorator(func: Callable) -> Callable:
        name = span_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            span = Tracer.start_span(name)
            
            if span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
            
            try:
                result = await func(*args, **kwargs)
                
                if span:
                    span.set_status(SpanStatus.OK)
                
                return result
            
            except Exception as e:
                if span:
                    span.set_status(SpanStatus.ERROR, str(e))
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                raise
            
            finally:
                if span:
                    Tracer.end_span(span)
        
        return wrapper
    
    return decorator


def trace_sync(span_name: Optional[str] = None):
    """Decorator to trace synchronous functions"""
    def decorator(func: Callable) -> Callable:
        name = span_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            span = Tracer.start_span(name)
            
            if span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
            
            try:
                result = func(*args, **kwargs)
                
                if span:
                    span.set_status(SpanStatus.OK)
                
                return result
            
            except Exception as e:
                if span:
                    span.set_status(SpanStatus.ERROR, str(e))
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                raise
            
            finally:
                if span:
                    Tracer.end_span(span)
        
        return wrapper
    
    return decorator


# ============================================================================
# Context Manager for Manual Tracing
# ============================================================================

class traced_span:
    """
    Context manager for manual span creation.
    
    Usage:
        async with traced_span("database.query") as span:
            span.set_attribute("query", sql)
            result = await db.execute(sql)
            span.set_attribute("rows_returned", len(result))
    """
    
    def __init__(self, name: str):
        self.name = name
        self.span: Optional[Span] = None
    
    def __enter__(self) -> Optional[Span]:
        self.span = Tracer.start_span(self.name)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.set_status(SpanStatus.ERROR, str(exc_val))
                self.span.set_attribute("error.type", exc_type.__name__)
            else:
                self.span.set_status(SpanStatus.OK)
            
            Tracer.end_span(self.span)
        
        return False  # Don't suppress exceptions
    
    async def __aenter__(self) -> Optional[Span]:
        self.span = Tracer.start_span(self.name)
        return self.span
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.set_status(SpanStatus.ERROR, str(exc_val))
                self.span.set_attribute("error.type", exc_type.__name__)
            else:
                self.span.set_status(SpanStatus.OK)
            
            Tracer.end_span(self.span)
        
        return False
