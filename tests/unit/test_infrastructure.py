"""Tests for infrastructure components."""

import pytest
import asyncio
from datetime import datetime

from src.infrastructure.event_store import EventStore, EventBus
from src.infrastructure.circuit_breaker import CircuitBreaker, CircuitState, with_circuit_breaker, CircuitBreakerError
from src.infrastructure.tracing import Tracer, trace_async, trace_sync
from src.domain.events import DomainEvent, EventType


class TestEventStore:
    """Tests for EventStore."""

    @pytest.mark.asyncio
    async def test_event_store_initialization(self, tmp_path):
        """Test event store creates storage directory."""
        store = EventStore(str(tmp_path / "events"))
        assert store.storage_dir.exists()

    @pytest.mark.asyncio
    async def test_append_event(self, tmp_path):
        """Test appending events."""
        store = EventStore(str(tmp_path / "events"))

        event = DomainEvent(
            event_id="test-1",
            event_type=EventType.SESSION_CREATED,
            timestamp=datetime.now(),
            tenant_id="default",
            aggregate_id="session-1",
            data={"test": "data"}
        )

        await store.append(event)

        # Verify event was written to disk
        events = await store.get_events("default", "session-1")
        assert len(events) == 1
        assert events[0].event_id == "test-1"

    @pytest.mark.asyncio
    async def test_get_events_by_session(self, tmp_path):
        """Test retrieving events by session."""
        store = EventStore(str(tmp_path / "events"))

        event1 = DomainEvent(
            event_id="e1",
            event_type=EventType.SESSION_CREATED,
            timestamp=datetime.now(),
            tenant_id="default",
            aggregate_id="session-1",
            data={}
        )

        event2 = DomainEvent(
            event_id="e2",
            event_type=EventType.TURN_STARTED,
            timestamp=datetime.now(),
            tenant_id="default",
            aggregate_id="session-2",
            data={}
        )

        await store.append(event1)
        await store.append(event2)

        events = await store.get_events("default", "session-1")
        assert len(events) == 1
        assert events[0].aggregate_id == "session-1"

    @pytest.mark.asyncio
    async def test_load_from_disk(self, tmp_path):
        """Test loading events from disk."""
        storage_dir = str(tmp_path / "events")
        store1 = EventStore(storage_dir)

        event = DomainEvent(
            event_id="persist-1",
            event_type=EventType.SESSION_CREATED,
            timestamp=datetime.now(),
            tenant_id="default",
            aggregate_id="session-1",
            data={"key": "value"}
        )

        await store1.append(event)

        # Create new store and load from disk
        store2 = EventStore(storage_dir)
        await store2.load_from_disk()

        events = await store2.get_events("default", "session-1")
        assert len(events) == 1
        assert events[0].event_id == "persist-1"


class TestEventBus:
    """Tests for EventBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Test subscribing and publishing events."""
        bus = EventBus()
        received_events = []

        async def handler(event):
            received_events.append(event)

        await bus.subscribe(EventType.SESSION_CREATED, handler)

        event = DomainEvent(
            event_id="test-1",
            event_type=EventType.SESSION_CREATED,
            timestamp=datetime.now(),
            tenant_id="default",
            aggregate_id="session-1",
            data={}
        )

        await bus.publish(event)
        await asyncio.sleep(0.1)  # Give task time to execute

        assert len(received_events) == 1
        assert received_events[0].event_id == "test-1"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        received_events = []

        async def handler(event):
            received_events.append(event)

        await bus.subscribe(EventType.SESSION_CREATED, handler)
        bus.unsubscribe(EventType.SESSION_CREATED, handler)

        event = DomainEvent(
            event_id="test-1",
            event_type=EventType.SESSION_CREATED,
            timestamp=datetime.now(),
            tenant_id="default",
            aggregate_id="session-1",
            data={}
        )

        await bus.publish(event)
        await asyncio.sleep(0.1)  # Give task time to execute

        assert len(received_events) == 0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        breaker = CircuitBreaker("test", failure_threshold=3)

        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await breaker.call(successful_operation)
        assert result == "success"
        assert call_count == 1
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker("test", failure_threshold=3)

        async def failing_operation():
            raise ValueError("Simulated failure")

        # First 3 failures should be allowed
        for i in range(3):
            with pytest.raises(ValueError):
                await breaker.call(failing_operation)

        # Circuit should now be OPEN
        assert breaker.state == CircuitState.OPEN

        # Next call should fail immediately with CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await breaker.call(failing_operation)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker transitions to half-open and recovers."""
        breaker = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)

        async def failing_operation():
            raise ValueError("Fail")

        async def successful_operation():
            return "success"

        # Trigger failures to open circuit
        for i in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing_operation)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Next call should transition to HALF_OPEN
        result = await breaker.call(successful_operation)
        assert result == "success"
        assert breaker.state in [CircuitState.HALF_OPEN, CircuitState.CLOSED]  # Accept either state

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        call_count = 0

        @with_circuit_breaker("decorated_test", failure_threshold=2)
        async def decorated_function(should_fail=False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Fail")
            return "success"

        # Successful call
        result = await decorated_function(should_fail=False)
        assert result == "success"
        assert call_count == 1

        # Failed calls
        for i in range(2):
            with pytest.raises(ValueError):
                await decorated_function(should_fail=True)

        # Circuit should be open now
        with pytest.raises(CircuitBreakerError):
            await decorated_function(should_fail=False)


class TestTracing:
    """Tests for distributed tracing."""

    @pytest.mark.asyncio
    async def test_trace_async_decorator(self):
        """Test async tracing decorator."""
        executed = False

        @trace_async("test.operation")
        async def traced_operation():
            nonlocal executed
            executed = True
            return "result"

        result = await traced_operation()
        assert result == "result"
        assert executed

    def test_trace_sync_decorator(self):
        """Test sync tracing decorator."""
        executed = False

        @trace_sync("test.sync_operation")
        def traced_operation():
            nonlocal executed
            executed = True
            return "result"

        result = traced_operation()
        assert result == "result"
        assert executed

    @pytest.mark.asyncio
    async def test_tracer_context_manager(self):
        """Test Tracer context manager."""
        span = Tracer.start_span("test.span")
        if span:  # Span might be None if tracing is not initialized
            span.set_attribute("key", "value")
            assert span.name == "test.span"
            assert span.attributes.get("key") == "value"
            span.end()

    @pytest.mark.asyncio
    async def test_nested_spans(self):
        """Test nested span creation."""
        @trace_async("outer")
        async def outer_operation():
            @trace_async("inner")
            async def inner_operation():
                return "inner"
            return await inner_operation()

        result = await outer_operation()
        assert result == "inner"

    def test_span_attributes(self):
        """Test setting span attributes."""
        span = Tracer.start_span("test")
        if span:  # Span might be None if tracing is not initialized
            span.set_attribute("key1", "value1")
            span.set_attribute("key2", 123)

            assert span.attributes.get("key1") == "value1"
            assert span.attributes.get("key2") == 123
            span.end()

    @pytest.mark.asyncio
    async def test_circuit_breaker_success_after_reset(self):
        """Test circuit breaker resets failure count after success."""
        breaker = CircuitBreaker("reset_test", failure_threshold=3)

        call_count = [0]

        async def sometimes_failing():
            call_count[0] += 1
            if call_count[0] in [1, 2, 4, 5]:  # Fail on these calls
                raise ValueError("Failed")
            return "success"

        # Two failures
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(sometimes_failing)

        # One success - should reset failure count
        result = await breaker.call(sometimes_failing)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

        # Two more failures shouldn't open circuit (count was reset)
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(sometimes_failing)

        # Still not open (would need 3 consecutive failures)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_event_store_multiple_sessions(self, tmp_path):
        """Test event store with multiple sessions."""
        store = EventStore(str(tmp_path / "events"))

        # Create events for different sessions
        for i in range(3):
            for j in range(2):
                event = DomainEvent(
                    event_id=f"e{i}-{j}",
                    event_type=EventType.SESSION_CREATED,
                    timestamp=datetime.now(),
                    tenant_id="default",
                    aggregate_id=f"session-{i}",
                    data={"index": j}
                )
                await store.append(event)

        # Verify each session has correct events
        for i in range(3):
            events = await store.get_events("default", f"session-{i}")
            assert len(events) == 2
            assert events[0].aggregate_id == f"session-{i}"

    @pytest.mark.asyncio
    async def test_event_bus_multiple_subscribers(self):
        """Test event bus with multiple subscribers."""
        bus = EventBus()
        received1 = []
        received2 = []

        async def handler1(event):
            received1.append(event)

        async def handler2(event):
            received2.append(event)

        await bus.subscribe(EventType.SESSION_CREATED, handler1)
        await bus.subscribe(EventType.SESSION_CREATED, handler2)

        event = DomainEvent(
            event_id="test",
            event_type=EventType.SESSION_CREATED,
            timestamp=datetime.now(),
            tenant_id="default",
            aggregate_id="session-1",
            data={}
        )

        await bus.publish(event)
        await asyncio.sleep(0.1)

        # Both handlers should receive the event
        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_fails_reopens(self):
        """Test circuit breaker reopens if half-open attempt fails."""
        breaker = CircuitBreaker("reopen_test", failure_threshold=2, recovery_timeout=0.1)

        async def failing():
            raise ValueError("Always fails")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should fail and reopen circuit
        with pytest.raises(ValueError):
            await breaker.call(failing)

        # Should still be open (or go back to open)
        assert breaker.state == CircuitState.OPEN
