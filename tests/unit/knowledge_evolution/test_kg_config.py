# tests/unit/knowledge_evolution/test_kg_config.py
import pytest
from homie_core.config import KnowledgeGraphConfig


class TestKnowledgeGraphConfig:
    def test_defaults(self):
        cfg = KnowledgeGraphConfig()
        assert cfg.enabled is True
        assert cfg.intake.surface_pass is True
        assert cfg.intake.deep_pass is True
        assert cfg.intake.deep_pass_top_percent == 20
        assert cfg.reasoning.entity_resolution is True
        assert cfg.reasoning.max_inference_hops == 2
        assert cfg.temporal.confidence_decay_rate == 0.99
