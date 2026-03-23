"""Priority-based task queue for agent work distribution."""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=False)
class Task:
    """A unit of work for an agent."""

    task_id: str
    payload: dict[str, Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)


class TaskQueue:
    """Thread-safe priority queue — higher priority tasks are dequeued first."""

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue[tuple[int, float, Task]] = (
            queue.PriorityQueue()
        )
        self._lock = threading.Lock()
        self._count = 0  # tiebreaker for equal priorities

    def enqueue(self, task: Task, priority: int | None = None) -> None:
        """Add a task.  *priority* overrides task.priority if given."""
        p = priority if priority is not None else task.priority
        with self._lock:
            self._count += 1
            # Negate so higher priority = dequeued first; count for FIFO tiebreak.
            self._queue.put((-p, self._count, task))

    def dequeue(self, timeout: float | None = None) -> Task:
        """Remove and return the highest-priority task.

        Raises ``queue.Empty`` if *timeout* expires or queue is empty and
        *timeout* is 0.
        """
        _, _, task = self._queue.get(timeout=timeout)
        return task

    def peek(self) -> Task | None:
        """Return the highest-priority task without removing it, or None."""
        with self._lock:
            if self._queue.empty():
                return None
            item = self._queue.queue[0]  # heap peek
            return item[2]

    def size(self) -> int:
        return self._queue.qsize()
