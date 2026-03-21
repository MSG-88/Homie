# tests/unit/self_healing/test_recovery_engine.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.engine import RecoveryEngine, RecoveryTier, RecoveryResult
from homie_core.self_healing.probes.base import HealthStatus


class TestRecoveryTier:
    def test_tier_ordering(self):
        assert RecoveryTier.RETRY < RecoveryTier.FALLBACK
        assert RecoveryTier.FALLBACK < RecoveryTier.REBUILD
        assert RecoveryTier.REBUILD < RecoveryTier.DEGRADE


class TestRecoveryEngine:
    def test_register_and_execute_strategy(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock())
        strategy = MagicMock(return_value=RecoveryResult(success=True, action="retried", tier=RecoveryTier.RETRY))
        engine.register_strategy("inference", RecoveryTier.RETRY, strategy)
        result = engine.recover("inference", HealthStatus.FAILED, error="timeout")
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY
        strategy.assert_called_once()

    def test_escalates_through_tiers(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock())
        t1 = MagicMock(return_value=RecoveryResult(success=False, action="retry failed", tier=RecoveryTier.RETRY))
        t2 = MagicMock(return_value=RecoveryResult(success=True, action="fallback ok", tier=RecoveryTier.FALLBACK))
        engine.register_strategy("inference", RecoveryTier.RETRY, t1)
        engine.register_strategy("inference", RecoveryTier.FALLBACK, t2)
        result = engine.recover("inference", HealthStatus.FAILED, error="timeout")
        assert result.success is True
        assert result.tier == RecoveryTier.FALLBACK
        t1.assert_called_once()
        t2.assert_called_once()

    def test_all_tiers_fail_returns_last_result(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock())
        t1 = MagicMock(return_value=RecoveryResult(success=False, action="retry fail", tier=RecoveryTier.RETRY))
        engine.register_strategy("inference", RecoveryTier.RETRY, t1)
        result = engine.recover("inference", HealthStatus.FAILED, error="fatal")
        assert result.success is False

    def test_no_strategy_registered_returns_failure(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock())
        result = engine.recover("unknown_module", HealthStatus.FAILED, error="no strategy")
        assert result.success is False

    def test_recovery_logged_to_event_bus(self):
        bus = MagicMock()
        log = MagicMock()
        engine = RecoveryEngine(event_bus=bus, health_log=log)
        strategy = MagicMock(return_value=RecoveryResult(success=True, action="ok", tier=RecoveryTier.RETRY))
        engine.register_strategy("storage", RecoveryTier.RETRY, strategy)
        engine.recover("storage", HealthStatus.FAILED, error="locked")
        bus.publish.assert_called()
        log.write.assert_called()

    def test_max_tier_respected(self):
        engine = RecoveryEngine(event_bus=MagicMock(), health_log=MagicMock(), max_tier=RecoveryTier.FALLBACK)
        t1 = MagicMock(return_value=RecoveryResult(success=False, action="fail", tier=RecoveryTier.RETRY))
        t2 = MagicMock(return_value=RecoveryResult(success=False, action="fail", tier=RecoveryTier.FALLBACK))
        t3 = MagicMock(return_value=RecoveryResult(success=True, action="ok", tier=RecoveryTier.REBUILD))
        engine.register_strategy("m", RecoveryTier.RETRY, t1)
        engine.register_strategy("m", RecoveryTier.FALLBACK, t2)
        engine.register_strategy("m", RecoveryTier.REBUILD, t3)
        result = engine.recover("m", HealthStatus.FAILED, error="err")
        assert result.success is False
        t3.assert_not_called()  # T3 not attempted — max_tier is FALLBACK
