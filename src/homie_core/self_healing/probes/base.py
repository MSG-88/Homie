"""Base probe class and health status types."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status ordered by severity for comparison."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

    # Severity ordering for comparisons
    _severity = None

    def __lt__(self, other):
        if not isinstance(other, HealthStatus):
            return NotImplemented
        order = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.FAILED, HealthStatus.UNKNOWN]
        return order.index(self) < order.index(other)

    def __le__(self, other):
        return self == other or self.__lt__(other)

    def __gt__(self, other):
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return not self.__le__(other)

    def __ge__(self, other):
        return self == other or self.__gt__(other)


@dataclass
class ProbeResult:
    """Result of a health probe check."""

    status: HealthStatus
    latency_ms: float
    error_count: int
    last_error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "metadata": self.metadata,
        }


class BaseProbe(ABC):
    """Base class for health probes. Subclasses implement check()."""

    name: str = "unnamed"
    interval: float = 30.0  # seconds between checks

    @abstractmethod
    def check(self) -> ProbeResult:
        """Run the health check. Override in subclasses."""
        ...

    def run(self) -> ProbeResult:
        """Run the probe with error handling and timing."""
        start = time.perf_counter()
        try:
            result = self.check()
            elapsed = (time.perf_counter() - start) * 1000
            result.latency_ms = elapsed
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Probe %s failed: %s", self.name, exc)
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=elapsed,
                error_count=1,
                last_error=str(exc),
            )
