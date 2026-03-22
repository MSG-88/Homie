# tests/unit/self_optimizer/test_config.py
import pytest
from homie_core.config import SelfOptimizerConfig


class TestSelfOptimizerConfig:
    def test_defaults(self):
        cfg = SelfOptimizerConfig()
        assert cfg.enabled is True
        assert cfg.prompt.deduplication is True
        assert cfg.model.auto_temperature is True
        assert cfg.pipeline.gating_enabled is True
        assert cfg.pipeline.promotion_threshold == 3
