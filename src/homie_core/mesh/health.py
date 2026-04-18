"""Mesh health checks — verify all subsystems are operational."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    healthy: bool
    latency_ms: float = 0.0
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Aggregated health of all mesh subsystems."""
    checks: list[HealthCheck] = field(default_factory=list)
    timestamp: str = ""

    @property
    def healthy(self) -> bool:
        return all(c.healthy for c in self.checks)

    @property
    def status(self) -> str:
        if not self.checks:
            return "unknown"
        if self.healthy:
            return "healthy"
        failed = [c.name for c in self.checks if not c.healthy]
        return f"degraded ({', '.join(failed)})"

    def summary(self) -> str:
        lines = [f"System: {self.status}"]
        for c in self.checks:
            icon = "+" if c.healthy else "X"
            lines.append(f"  [{icon}] {c.name}: {c.message} ({c.latency_ms:.0f}ms)")
        return "\n".join(lines)


class MeshHealthChecker:
    """Runs health checks against all mesh subsystems."""

    def __init__(self, mesh_context=None):
        self._ctx = mesh_context

    def run_all(self) -> SystemHealth:
        """Run all health checks and return aggregated result."""
        from homie_core.utils import utc_now

        checks = []
        checks.append(self._check_identity())
        checks.append(self._check_event_store())
        checks.append(self._check_capabilities())
        checks.append(self._check_feedback_store())
        checks.append(self._check_auth_store())
        checks.append(self._check_registry())

        return SystemHealth(checks=checks, timestamp=utc_now().isoformat())

    def _timed_check(self, name: str, fn) -> HealthCheck:
        """Run a check function with timing."""
        start = time.monotonic()
        try:
            result = fn()
            latency = (time.monotonic() - start) * 1000
            return HealthCheck(name=name, healthy=True, latency_ms=latency,
                             message=result.get("message", "ok"),
                             details=result)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return HealthCheck(name=name, healthy=False, latency_ms=latency,
                             message=str(e))

    def _check_identity(self) -> HealthCheck:
        def check():
            if self._ctx is None:
                raise RuntimeError("No mesh context")
            return {"message": f"node={self._ctx.node_name}", "node_id": self._ctx.node_id}
        return self._timed_check("identity", check)

    def _check_event_store(self) -> HealthCheck:
        def check():
            if self._ctx is None:
                raise RuntimeError("No mesh context")
            count = self._ctx.mesh_manager.event_count()
            return {"message": f"{count} events", "count": count}
        return self._timed_check("event_store", check)

    def _check_capabilities(self) -> HealthCheck:
        def check():
            if self._ctx is None:
                raise RuntimeError("No mesh context")
            score = self._ctx.capabilities.capability_score()
            return {"message": f"score={score:.0f}", "score": score}
        return self._timed_check("capabilities", check)

    def _check_feedback_store(self) -> HealthCheck:
        def check():
            if self._ctx is None:
                raise RuntimeError("No mesh context")
            count = self._ctx.feedback_store.total_count()
            return {"message": f"{count} signals", "count": count}
        return self._timed_check("feedback_store", check)

    def _check_auth_store(self) -> HealthCheck:
        def check():
            if self._ctx is None:
                raise RuntimeError("No mesh context")
            users = len(self._ctx.auth_store.list_users())
            return {"message": f"{users} users", "count": users}
        return self._timed_check("auth_store", check)

    def _check_registry(self) -> HealthCheck:
        def check():
            if self._ctx is None:
                raise RuntimeError("No mesh context")
            nodes = len(self._ctx.registry.list_all())
            return {"message": f"{nodes} peers", "count": nodes}
        return self._timed_check("registry", check)
