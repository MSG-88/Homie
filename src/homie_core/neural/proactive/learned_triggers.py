"""Learned trigger manager — observes user requests and creates proactive tasks.

When a user repeatedly asks for the same kind of analysis or report, Homie
learns to produce it proactively. Feeds from the Adaptive Learning
ObservationStream.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from homie_core.neural.proactive.trigger_engine import ProactiveTask

logger = logging.getLogger(__name__)

# Default interval for learned schedule-based tasks (weekly)
_DEFAULT_LEARNED_INTERVAL = 7 * 24 * 3600

# How many times a request pattern must be seen before learning it
_LEARNING_THRESHOLD = 2


class LearnedTriggerManager:
    """Learns proactive tasks from repeated user requests.

    *storage* must support:
    - ``get(key, default=None)`` -> value
    - ``set(key, value)``

    This is compatible with Homie's existing storage backends (dict-like).
    """

    def __init__(self, storage: Any) -> None:
        self._storage = storage
        self._request_counts: dict[str, int] = self._load("learned_request_counts", {})
        self._learned_tasks: dict[str, dict] = self._load("learned_tasks", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn_from_request(self, user_request: str, domain: str) -> None:
        """Observe a user request and potentially create a proactive task.

        The request text is normalised into a pattern key.  After the pattern
        is seen ``_LEARNING_THRESHOLD`` times, a proactive task is created.
        """
        if not user_request or not user_request.strip():
            return

        pattern = self._normalise(user_request)
        if not pattern:
            return

        self._request_counts[pattern] = self._request_counts.get(pattern, 0) + 1
        self._save("learned_request_counts", self._request_counts)

        if (
            self._request_counts[pattern] >= _LEARNING_THRESHOLD
            and pattern not in self._learned_tasks
        ):
            task_data = {
                "id": f"learned_{uuid.uuid4().hex[:8]}",
                "trigger_type": "schedule",
                "trigger_config": {"interval_seconds": _DEFAULT_LEARNED_INTERVAL},
                "action": f"auto_{pattern}",
                "domain": domain,
                "priority": 6,
                "enabled": True,
                "source_request": user_request,
                "learned_at": time.time(),
            }
            self._learned_tasks[pattern] = task_data
            self._save("learned_tasks", self._learned_tasks)
            logger.info(
                "Learned new proactive task from repeated request: %s (domain=%s)",
                pattern,
                domain,
            )

    def get_learned_tasks(self) -> list[ProactiveTask]:
        """Return all learned proactive tasks."""
        tasks: list[ProactiveTask] = []
        for data in self._learned_tasks.values():
            if not data.get("enabled", True):
                continue
            tasks.append(
                ProactiveTask(
                    id=data["id"],
                    trigger_type=data["trigger_type"],
                    trigger_config=data["trigger_config"],
                    action=data["action"],
                    domain=data["domain"],
                    priority=data.get("priority", 6),
                    last_run=data.get("last_run"),
                    enabled=data.get("enabled", True),
                )
            )
        return tasks

    def get_request_counts(self) -> dict[str, int]:
        """Return current request pattern counts (for debugging/testing)."""
        return dict(self._request_counts)

    def remove_learned_task(self, pattern: str) -> bool:
        """Remove a learned task by its normalised pattern key."""
        removed = self._learned_tasks.pop(pattern, None) is not None
        if removed:
            self._save("learned_tasks", self._learned_tasks)
        return removed

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _load(self, key: str, default: Any) -> Any:
        try:
            val = self._storage.get(key, default)
            if isinstance(val, str):
                return json.loads(val)
            return val if val is not None else default
        except Exception:
            return default

    def _save(self, key: str, value: Any) -> None:
        try:
            self._storage.set(key, json.dumps(value, default=str))
        except Exception:
            logger.warning("Failed to persist %s to storage", key)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(request: str) -> str:
        """Normalise a user request into a short pattern key.

        Strips common question words, lowercases, and truncates to keep
        storage keys manageable.
        """
        text = request.lower().strip()
        # Remove common prefixes
        for prefix in (
            "can you ", "please ", "could you ", "i need ", "i want ",
            "give me ", "show me ", "generate ", "create ", "make ",
        ):
            if text.startswith(prefix):
                text = text[len(prefix):]
        # Remove trailing punctuation
        text = text.rstrip("?.! ")
        # Collapse whitespace and truncate
        words = text.split()
        return "_".join(words[:8])
