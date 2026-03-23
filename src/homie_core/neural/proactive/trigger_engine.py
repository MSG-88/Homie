"""Proactive trigger engine — monitors conditions and fires tasks autonomously.

Supports schedule-based, event-based, threshold-based, and pattern-based
triggers that drive Homie's proactive intelligence.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProactiveTask:
    """A proactive task that fires when its trigger condition is met."""

    id: str
    trigger_type: str  # "schedule", "event", "threshold", "pattern"
    trigger_config: dict
    action: str
    domain: str
    priority: int = 5
    last_run: Optional[float] = None
    enabled: bool = True


# ---------------------------------------------------------------------------
# Trigger checking helpers
# ---------------------------------------------------------------------------

def _check_schedule_trigger(task: ProactiveTask, state: dict, now: float) -> bool:
    """Check if a schedule-based trigger should fire.

    trigger_config keys:
    - interval_seconds: fire every N seconds
    - cron_hour / cron_day_of_week / cron_day_of_month: simplified cron
    """
    cfg = task.trigger_config
    interval = cfg.get("interval_seconds")
    if interval is not None:
        if task.last_run is None:
            return True
        return (now - task.last_run) >= interval
    return False


def _check_event_trigger(task: ProactiveTask, state: dict) -> bool:
    """Check if an event-based trigger should fire.

    trigger_config keys:
    - event_type: the event key to watch in state["events"]
    """
    cfg = task.trigger_config
    event_type = cfg.get("event_type", "")
    events = state.get("events", [])
    if isinstance(events, list):
        return any(
            (e.get("type") == event_type if isinstance(e, dict) else e == event_type)
            for e in events
        )
    return False


def _check_threshold_trigger(task: ProactiveTask, state: dict) -> bool:
    """Check if a threshold-based trigger should fire.

    trigger_config keys:
    - metric: key in state["metrics"]
    - operator: "gt", "lt", "gte", "lte", "eq"
    - value: threshold value
    """
    cfg = task.trigger_config
    metric_key = cfg.get("metric", "")
    operator = cfg.get("operator", "gt")
    threshold = cfg.get("value", 0)

    metrics = state.get("metrics", {})
    actual = metrics.get(metric_key)
    if actual is None:
        return False

    ops = {
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
        "eq": lambda a, b: a == b,
    }
    check = ops.get(operator, ops["gt"])
    return check(actual, threshold)


def _check_pattern_trigger(task: ProactiveTask, state: dict) -> bool:
    """Check if a pattern-based trigger should fire.

    trigger_config keys:
    - pattern_key: key in state["patterns"] that should be truthy
    """
    cfg = task.trigger_config
    pattern_key = cfg.get("pattern_key", "")
    patterns = state.get("patterns", {})
    return bool(patterns.get(pattern_key))


_TRIGGER_CHECKERS = {
    "schedule": _check_schedule_trigger,
    "event": _check_event_trigger,
    "threshold": _check_threshold_trigger,
    "pattern": _check_pattern_trigger,
}


class TriggerEngine:
    """Manages proactive tasks and checks their triggers against current state."""

    def __init__(self) -> None:
        self._tasks: dict[str, ProactiveTask] = {}

    @property
    def tasks(self) -> list[ProactiveTask]:
        """All registered tasks."""
        return list(self._tasks.values())

    def register_task(self, task: ProactiveTask) -> None:
        """Register a proactive task."""
        self._tasks[task.id] = task
        logger.debug("Registered proactive task: %s (%s)", task.id, task.action)

    def unregister_task(self, task_id: str) -> bool:
        """Remove a task by id. Returns True if removed."""
        return self._tasks.pop(task_id, None) is not None

    def check_triggers(self, current_state: dict) -> list[ProactiveTask]:
        """Check all enabled tasks and return those whose triggers fired."""
        now = current_state.get("timestamp", time.time())
        triggered: list[ProactiveTask] = []

        for task in self._tasks.values():
            if not task.enabled:
                continue

            checker = _TRIGGER_CHECKERS.get(task.trigger_type)
            if checker is None:
                logger.warning("Unknown trigger type: %s", task.trigger_type)
                continue

            try:
                # Schedule checker needs the timestamp
                if task.trigger_type == "schedule":
                    fired = checker(task, current_state, now)
                else:
                    fired = checker(task, current_state)

                if fired:
                    task.last_run = now
                    triggered.append(task)
            except Exception:
                logger.exception("Error checking trigger for task %s", task.id)

        # Sort by priority (lower number = higher priority)
        triggered.sort(key=lambda t: t.priority)
        return triggered

    def get_default_tasks(self) -> list[ProactiveTask]:
        """Return built-in proactive tasks matching the spec triggers."""
        defaults = [
            ProactiveTask(
                id="monthly_financial_summary",
                trigger_type="schedule",
                trigger_config={"interval_seconds": 30 * 24 * 3600},  # ~monthly
                action="generate_financial_summary",
                domain="finance",
                priority=3,
            ),
            ProactiveTask(
                id="weekly_work_summary",
                trigger_type="schedule",
                trigger_config={"interval_seconds": 7 * 24 * 3600},  # weekly
                action="generate_work_summary",
                domain="general",
                priority=4,
            ),
            ProactiveTask(
                id="daily_morning_briefing",
                trigger_type="schedule",
                trigger_config={"interval_seconds": 24 * 3600},  # daily
                action="generate_morning_briefing",
                domain="general",
                priority=2,
            ),
            ProactiveTask(
                id="new_bank_statement",
                trigger_type="event",
                trigger_config={"event_type": "bank_statement_received"},
                action="categorize_transactions",
                domain="accounting",
                priority=3,
            ),
            ProactiveTask(
                id="new_invoice",
                trigger_type="event",
                trigger_config={"event_type": "invoice_received"},
                action="process_invoice",
                domain="accounting",
                priority=4,
            ),
            ProactiveTask(
                id="contract_deadline",
                trigger_type="event",
                trigger_config={"event_type": "contract_deadline_approaching"},
                action="alert_contract_deadline",
                domain="legal",
                priority=1,
            ),
            ProactiveTask(
                id="unusual_transaction",
                trigger_type="threshold",
                trigger_config={"metric": "transaction_anomaly_score", "operator": "gt", "value": 0.8},
                action="alert_unusual_transaction",
                domain="finance",
                priority=1,
            ),
            ProactiveTask(
                id="budget_overrun",
                trigger_type="threshold",
                trigger_config={"metric": "budget_utilization", "operator": "gt", "value": 0.9},
                action="alert_budget_overrun",
                domain="finance",
                priority=2,
            ),
            ProactiveTask(
                id="tax_filing_deadline",
                trigger_type="event",
                trigger_config={"event_type": "tax_deadline_approaching"},
                action="prepare_tax_summary",
                domain="tax",
                priority=1,
            ),
            ProactiveTask(
                id="important_email",
                trigger_type="pattern",
                trigger_config={"pattern_key": "important_email_detected"},
                action="summarize_important_email",
                domain="general",
                priority=2,
            ),
        ]
        return defaults
