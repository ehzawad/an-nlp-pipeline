"""
Event Store - Immutable append-only log of all domain events.

Provides:
- Event persistence
- Event replay for state reconstruction
- Event streaming for subscribers
- Audit trail
"""

import json
from typing import List, Optional, AsyncIterator
from datetime import datetime
from pathlib import Path
import aiofiles
import asyncio
from collections import defaultdict

from src.domain.events import DomainEvent, EventType


class EventStore:
    """
    Append-only event store with async support.
    
    Production implementation would use:
    - PostgreSQL with JSONB (transactional)
    - EventStoreDB (purpose-built for event sourcing)
    - Kafka (distributed streaming)
    
    This implementation uses async file I/O for demo purposes.
    """
    
    def __init__(self, storage_dir: str = "./event_store"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        
        # In-memory index for fast lookups (would be Redis/PostgreSQL in prod)
        self._index: dict = defaultdict(list)
    
    async def append(self, event: DomainEvent) -> None:
        """
        Append event to store (immutable).
        
        Thread-safe with async lock.
        """
        async with self._lock:
            # Write to append-only log
            log_file = self._get_log_file(event.tenant_id, event.aggregate_id)
            
            async with aiofiles.open(log_file, mode='a') as f:
                await f.write(json.dumps(event.to_dict()) + "\n")
            
            # Update index
            self._index[f"{event.tenant_id}:{event.aggregate_id}"].append(event)
    
    async def append_batch(self, events: List[DomainEvent]) -> None:
        """Append multiple events atomically"""
        async with self._lock:
            for event in events:
                await self.append(event)
    
    async def get_events(
        self,
        tenant_id: str,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[DomainEvent]:
        """
        Get all events for an aggregate (session).
        
        Used for event replay to reconstruct state.
        """
        key = f"{tenant_id}:{aggregate_id}"
        events = self._index.get(key, [])
        
        if to_version:
            events = events[from_version:to_version]
        else:
            events = events[from_version:]
        
        return events
    
    async def stream_events(
        self,
        tenant_id: str,
        aggregate_id: str,
        from_version: int = 0
    ) -> AsyncIterator[DomainEvent]:
        """Stream events as async iterator"""
        events = await self.get_events(tenant_id, aggregate_id, from_version)
        for event in events:
            yield event
    
    async def get_events_by_type(
        self,
        tenant_id: str,
        event_type: EventType,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None
    ) -> List[DomainEvent]:
        """Query events by type (for analytics)"""
        # In production, this would be an indexed query
        all_events = []
        for key, events in self._index.items():
            if key.startswith(f"{tenant_id}:"):
                all_events.extend(events)
        
        filtered = [
            e for e in all_events
            if e.event_type == event_type
        ]
        
        if from_timestamp:
            filtered = [e for e in filtered if e.timestamp >= from_timestamp]
        if to_timestamp:
            filtered = [e for e in filtered if e.timestamp <= to_timestamp]
        
        return filtered
    
    def _get_log_file(self, tenant_id: str, aggregate_id: str) -> Path:
        """Get log file path for aggregate"""
        tenant_dir = self.storage_dir / tenant_id
        tenant_dir.mkdir(exist_ok=True)
        return tenant_dir / f"{aggregate_id}.jsonl"
    
    async def load_from_disk(self) -> None:
        """Load events from disk into memory index (startup)"""
        for tenant_dir in self.storage_dir.iterdir():
            if not tenant_dir.is_dir():
                continue
            
            tenant_id = tenant_dir.name
            
            for log_file in tenant_dir.glob("*.jsonl"):
                aggregate_id = log_file.stem
                
                async with aiofiles.open(log_file, mode='r') as f:
                    async for line in f:
                        if line.strip():
                            event_dict = json.loads(line)
                            event = DomainEvent.from_dict(event_dict)
                            key = f"{tenant_id}:{aggregate_id}"
                            self._index[key].append(event)


class EventBus:
    """
    In-memory event bus for pub/sub.
    
    Production implementation would use:
    - RabbitMQ
    - Kafka
    - AWS SNS/SQS
    - Redis Pub/Sub
    """
    
    def __init__(self):
        self._subscribers: dict = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish event to all subscribers"""
        async with self._lock:
            # Notify subscribers of specific event type
            for handler in self._subscribers.get(event.event_type, []):
                asyncio.create_task(handler(event))
            
            # Notify wildcard subscribers (listen to all events)
            for handler in self._subscribers.get("*", []):
                asyncio.create_task(handler(event))
    
    async def subscribe(
        self,
        event_type: EventType,
        handler: callable
    ) -> None:
        """Subscribe to event type"""
        async with self._lock:
            self._subscribers[event_type].append(handler)
    
    async def subscribe_all(self, handler: callable) -> None:
        """Subscribe to all events (wildcard)"""
        async with self._lock:
            self._subscribers["*"].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: callable) -> None:
        """Unsubscribe handler from event type"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)
