"""WebSocket event system for real-time UI communication.

Phase 5 - Real-time event streaming between backend and UI.
"""

import json
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from queue import Queue, Empty

logger = logging.getLogger(__name__)


@dataclass
class WSEvent:
    """WebSocket event."""
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        })


class EventBus:
    """Central event bus for real-time communication.

    Manages subscriptions and broadcasts events to all connected clients.
    """

    def __init__(self, max_history: int = 500):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._active = True

    def subscribe(self, event_type: str, callback: Callable[[WSEvent], None]):
        """Subscribe to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[WSEvent], None]):
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb != callback
                ]

    def publish(self, event: WSEvent):
        """Publish an event to all subscribers."""
        with self._lock:
            self._history.append(event)

        subscribers = []
        with self._lock:
            subscribers.extend(self._subscribers.get(event.event_type, []))
            subscribers.extend(self._subscribers.get("*", []))

        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Event subscriber error: {e}")

    def emit(self, event_type: str, data: Dict[str, Any] = None, session_id: str = ""):
        """Convenience method to emit an event."""
        event = WSEvent(
            event_type=event_type,
            data=data or {},
            session_id=session_id,
        )
        self.publish(event)

    def get_history(self, event_type: Optional[str] = None, limit: int = 50) -> List[WSEvent]:
        """Get event history."""
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def shutdown(self):
        """Shutdown the event bus."""
        self._active = False
        with self._lock:
            self._subscribers.clear()


class SSEHandler:
    """Server-Sent Events handler for HTTP streaming."""

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._client_queues: Dict[str, Queue] = {}
        self._lock = threading.Lock()

        def _on_event(event: WSEvent):
            self._broadcast(event)

        event_bus.subscribe("*", _on_event)

    def create_client(self, client_id: str) -> Queue:
        """Create a queue for a new SSE client."""
        with self._lock:
            q = Queue(maxsize=200)
            self._client_queues[client_id] = q
            return q

    def remove_client(self, client_id: str):
        """Remove an SSE client."""
        with self._lock:
            self._client_queues.pop(client_id, None)

    def _broadcast(self, event: WSEvent):
        """Broadcast event to all connected clients."""
        with self._lock:
            clients = dict(self._client_queues)

        dead = []
        for cid, q in clients.items():
            try:
                q.put_nowait(event.to_json())
            except Exception:
                dead.append(cid)

        for cid in dead:
            with self._lock:
                self._client_queues.pop(cid, None)

    def get_events_for_client(self, client_id: str, timeout: float = 1.0) -> Optional[str]:
        """Get next event for a client (blocking with timeout)."""
        with self._lock:
            q = self._client_queues.get(client_id)
        if not q:
            return None
        try:
            return q.get(timeout=timeout)
        except Empty:
            return None
