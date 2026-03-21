"""In-process event bus for health event pub/sub."""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class HealthEvent:
    """A health event published by a module or the watchdog."""

    module: str
    event_type: str  # probe_result, anomaly, recovery, improvement, rollback
    severity: str    # info, warning, error, critical
    details: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    version_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "event_type": self.event_type,
            "severity": self.severity,
            "details": self.details,
            "timestamp": self.timestamp,
            "version_id": self.version_id,
        }


EventCallback = Callable[[HealthEvent], None]


class EventBus:
    """Lightweight in-process pub/sub for health events."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventCallback]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[HealthEvent | None] = queue.Queue()
        self._running = True
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Subscribe to events. Use '*' to receive all events."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Remove a subscription."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass

    def publish(self, event: HealthEvent) -> None:
        """Publish an event to all matching subscribers."""
        if self._running:
            self._queue.put(event)

    def shutdown(self) -> None:
        """Stop the event bus."""
        self._running = False
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    def _process_loop(self) -> None:
        """Background worker that dispatches events to subscribers."""
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event is None:
                break

            with self._lock:
                callbacks = list(self._subscribers.get(event.event_type, []))
                callbacks.extend(self._subscribers.get("*", []))

            for cb in callbacks:
                try:
                    cb(event)
                except Exception:
                    logger.exception("Event handler failed for %s", event.event_type)
