"""Inter-agent message bus for the Neural Reasoning Engine.

Extends the EventBus pattern from self_healing with agent-aware routing:
subscribe by agent name or message type, priority-based dispatch.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """A message exchanged between agents on the bus."""

    from_agent: str
    to_agent: str
    message_type: str  # goal, result, query, status, error
    content: dict[str, Any]
    priority: int = 0
    parent_goal_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "parent_goal_id": self.parent_goal_id,
            "timestamp": self.timestamp,
        }


AgentCallback = Callable[[AgentMessage], None]


class AgentBus:
    """Thread-safe inter-agent communication bus with background dispatch.

    Subscribers can register by:
    - agent name  (receives messages addressed to that agent)
    - message type (receives all messages of that type, e.g. "goal")
    - wildcard "*" (receives every message)
    """

    def __init__(self) -> None:
        self._agent_subscribers: dict[str, list[AgentCallback]] = {}
        self._type_subscribers: dict[str, list[AgentCallback]] = {}
        self._lock = threading.Lock()
        self._queue: queue.PriorityQueue[tuple[int, int, AgentMessage | None]] = (
            queue.PriorityQueue()
        )
        self._seq = 0  # monotonic tiebreaker so messages with equal priority are FIFO
        self._running = True
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    # ── subscription ────────────────────────────────────────────────

    def subscribe_agent(self, agent_name: str, callback: AgentCallback) -> None:
        """Subscribe to messages addressed to *agent_name*."""
        with self._lock:
            self._agent_subscribers.setdefault(agent_name, []).append(callback)

    def subscribe_type(self, message_type: str, callback: AgentCallback) -> None:
        """Subscribe to all messages of *message_type*.  Use '*' for all."""
        with self._lock:
            self._type_subscribers.setdefault(message_type, []).append(callback)

    def unsubscribe_agent(self, agent_name: str, callback: AgentCallback) -> None:
        with self._lock:
            if agent_name in self._agent_subscribers:
                try:
                    self._agent_subscribers[agent_name].remove(callback)
                except ValueError:
                    pass

    def unsubscribe_type(self, message_type: str, callback: AgentCallback) -> None:
        with self._lock:
            if message_type in self._type_subscribers:
                try:
                    self._type_subscribers[message_type].remove(callback)
                except ValueError:
                    pass

    # ── publishing ──────────────────────────────────────────────────

    def publish(self, message: AgentMessage) -> None:
        """Enqueue a message for async dispatch (higher priority = dispatched first)."""
        if self._running:
            # PriorityQueue is a min-heap; negate priority so higher values go first.
            # Use a monotonic sequence number as tiebreaker (avoids comparing AgentMessage).
            with self._lock:
                self._seq += 1
                seq = self._seq
            self._queue.put((-message.priority, seq, message))

    # ── lifecycle ───────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._running = False
        self._queue.put((0, 0, None))  # sentinel
        self._worker.join(timeout=2.0)

    # ── internals ───────────────────────────────────────────────────

    def _process_loop(self) -> None:
        """Background worker — dispatches messages to matching subscribers."""
        while self._running:
            try:
                _neg_pri, _seq, message = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if message is None:
                break

            with self._lock:
                callbacks: list[AgentCallback] = []
                # 1) agent-targeted
                callbacks.extend(self._agent_subscribers.get(message.to_agent, []))
                # 2) message-type
                callbacks.extend(self._type_subscribers.get(message.message_type, []))
                # 3) wildcard
                callbacks.extend(self._type_subscribers.get("*", []))

            for cb in callbacks:
                try:
                    cb(message)
                except Exception:
                    logger.exception(
                        "AgentBus handler failed for %s -> %s",
                        message.from_agent,
                        message.to_agent,
                    )
