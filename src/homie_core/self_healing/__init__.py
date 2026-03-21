"""Homie Self-Healing Runtime — autonomous failure recovery and self-improvement."""

from .event_bus import EventBus, HealthEvent
from .guardian import Guardian
from .health_log import HealthLog
from .metrics import AnomalyAlert, MetricsCollector
from .resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ErrorCategory,
    classify_exception,
    resilient,
    retry_with_backoff,
    run_with_timeout,
)
from .watchdog import HealthWatchdog

__all__ = [
    # Core
    "HealthWatchdog",
    "EventBus",
    "HealthEvent",
    "HealthLog",
    "Guardian",
    # Metrics
    "MetricsCollector",
    "AnomalyAlert",
    # Resilience
    "resilient",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "ErrorCategory",
    "classify_exception",
    "retry_with_backoff",
    "run_with_timeout",
]
