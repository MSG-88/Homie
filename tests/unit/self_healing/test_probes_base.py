# tests/unit/self_healing/test_probes_base.py
import pytest
from homie_core.self_healing.probes.base import BaseProbe, HealthStatus, ProbeResult


class TestHealthStatus:
    def test_healthy(self):
        s = HealthStatus.HEALTHY
        assert s.value == "healthy"

    def test_ordering(self):
        assert HealthStatus.HEALTHY < HealthStatus.DEGRADED
        assert HealthStatus.DEGRADED < HealthStatus.FAILED


class TestProbeResult:
    def test_creation(self):
        r = ProbeResult(
            status=HealthStatus.HEALTHY,
            latency_ms=15.5,
            error_count=0,
            metadata={"model": "qwen"},
        )
        assert r.status == HealthStatus.HEALTHY
        assert r.latency_ms == 15.5

    def test_to_dict(self):
        r = ProbeResult(status=HealthStatus.FAILED, latency_ms=0, error_count=3, last_error="timeout")
        d = r.to_dict()
        assert d["status"] == "failed"
        assert d["error_count"] == 3
        assert d["last_error"] == "timeout"


class ConcreteProbe(BaseProbe):
    name = "test_probe"
    interval = 10.0

    def check(self) -> ProbeResult:
        return ProbeResult(status=HealthStatus.HEALTHY, latency_ms=1.0, error_count=0)


class FailingProbe(BaseProbe):
    name = "failing_probe"
    interval = 10.0

    def check(self) -> ProbeResult:
        raise RuntimeError("probe crash")


class TestBaseProbe:
    def test_concrete_probe_works(self):
        probe = ConcreteProbe()
        result = probe.run()
        assert result.status == HealthStatus.HEALTHY

    def test_failing_probe_returns_failed_status(self):
        probe = FailingProbe()
        result = probe.run()
        assert result.status == HealthStatus.FAILED
        assert "probe crash" in result.last_error

    def test_probe_has_name_and_interval(self):
        probe = ConcreteProbe()
        assert probe.name == "test_probe"
        assert probe.interval == 10.0
