"""Recovery engine — orchestrates tiered recovery strategies."""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Optional

from ..event_bus import EventBus, HealthEvent
from ..health_log import HealthLog
from ..probes.base import HealthStatus

logger = logging.getLogger(__name__)


class RecoveryTier(IntEnum):
    RETRY = 1
    FALLBACK = 2
    REBUILD = 3
    DEGRADE = 4


@dataclass
class RecoveryResult:
    success: bool
    action: str
    tier: RecoveryTier
    details: dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# Strategy callable: (module, status, error, **context) -> RecoveryResult
RecoveryStrategy = Callable[..., RecoveryResult]


class RecoveryEngine:
    """Orchestrates tiered recovery — tries lightest fix first, escalates on failure."""

    def __init__(
        self,
        event_bus: EventBus,
        health_log: HealthLog,
        max_tier: RecoveryTier = RecoveryTier.DEGRADE,
    ) -> None:
        self._bus = event_bus
        self._log = health_log
        self._max_tier = max_tier
        # {module: {tier: strategy}}
        self._strategies: dict[str, dict[RecoveryTier, RecoveryStrategy]] = {}

    def register_strategy(
        self, module: str, tier: RecoveryTier, strategy: RecoveryStrategy
    ) -> None:
        """Register a recovery strategy for a module at a specific tier."""
        self._strategies.setdefault(module, {})[tier] = strategy

    def recover(
        self,
        module: str,
        status: HealthStatus,
        error: str = "",
        **context: Any,
    ) -> RecoveryResult:
        """Attempt recovery for a module by escalating through tiers."""
        strategies = self._strategies.get(module, {})

        if not strategies:
            logger.warning("No recovery strategies for module: %s", module)
            return RecoveryResult(
                success=False,
                action="no_strategy",
                tier=RecoveryTier.RETRY,
                details={"error": f"No recovery strategies registered for {module}"},
            )

        last_result = None

        for tier in sorted(RecoveryTier):
            if tier > self._max_tier:
                break

            strategy = strategies.get(tier)
            if strategy is None:
                continue

            logger.info("Attempting T%d recovery for %s: %s", tier, module, error)

            try:
                result = strategy(module=module, status=status, error=error, **context)
            except Exception as exc:
                logger.error("Recovery strategy T%d for %s crashed: %s", tier, module, exc)
                result = RecoveryResult(
                    success=False,
                    action=f"strategy_crash: {exc}",
                    tier=tier,
                )

            # Log the attempt
            severity = "info" if result.success else "warning"
            event = HealthEvent(
                module=module,
                event_type="recovery",
                severity=severity,
                details={
                    "tier": tier.value,
                    "action": result.action,
                    "success": result.success,
                    "error": error,
                },
            )
            self._bus.publish(event)
            self._log.write(event)

            if result.success:
                return result

            last_result = result

        return last_result or RecoveryResult(
            success=False,
            action="all_tiers_exhausted",
            tier=RecoveryTier.DEGRADE,
        )
