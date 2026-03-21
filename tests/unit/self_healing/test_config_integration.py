# tests/unit/self_healing/test_config_integration.py
import pytest
from homie_core.config import SelfHealingConfig, ImprovementConfig, RecoveryConfig


class TestSelfHealingConfig:
    def test_defaults(self):
        cfg = SelfHealingConfig()
        assert cfg.enabled is True
        assert cfg.probe_interval == 30
        assert cfg.critical_probe_interval == 10

    def test_improvement_defaults(self):
        cfg = ImprovementConfig()
        assert cfg.enabled is True
        assert cfg.max_mutations_per_day == 10
        assert cfg.monitoring_window == 300
        assert cfg.rollback_error_threshold == 0.20
        assert cfg.rollback_latency_threshold == 0.50

    def test_recovery_defaults(self):
        cfg = RecoveryConfig()
        assert cfg.max_tier == 4
        assert cfg.preemptive is True
        assert cfg.pattern_threshold == 3

    def test_core_lock_defaults(self):
        cfg = SelfHealingConfig()
        assert "self_healing/improvement/rollback.py" in cfg.core_lock
        assert "self_healing/guardian.py" in cfg.core_lock
        assert "security/" in cfg.core_lock
