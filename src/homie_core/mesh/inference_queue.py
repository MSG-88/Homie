"""Inference queue with priority scheduling for Hub inference server."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional


class InferencePriority:
    IMMEDIATE = 0
    BACKGROUND = 1
    BATCH = 2


@dataclass(order=True)
class InferenceRequest:
    priority: int = field(compare=True)
    _order: int = field(default=0, compare=True, repr=False)
    request_id: str = field(default="", compare=False)
    node_id: str = field(default="", compare=False)
    prompt: str = field(default="", compare=False)
    max_tokens: int = field(default=1024, compare=False)
    temperature: float = field(default=0.7, compare=False)
    stop: Optional[list[str]] = field(default=None, compare=False)


class InferenceQueue:
    """Thread-safe priority queue for inference requests."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._max_concurrent = max_concurrent
        self._active: dict[str, bool] = {}
        self._completed_count = 0
        self._order_counter = 0
        self._lock = threading.Lock()

    def submit(self, request: InferenceRequest) -> None:
        with self._lock:
            request._order = self._order_counter
            self._order_counter += 1
        self._queue.put(request)

    def get(self, timeout: float = 5.0) -> Optional[InferenceRequest]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def mark_active(self, request_id: str) -> None:
        with self._lock:
            self._active[request_id] = True

    def mark_done(self, request_id: str) -> None:
        with self._lock:
            self._active.pop(request_id, None)
            self._completed_count += 1

    def can_accept(self) -> bool:
        with self._lock:
            return len(self._active) < self._max_concurrent

    def pending_count(self) -> int:
        return self._queue.qsize()

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def stats(self) -> dict:
        with self._lock:
            return {
                "pending": self._queue.qsize(),
                "active": len(self._active),
                "completed": self._completed_count,
                "max_concurrent": self._max_concurrent,
            }
