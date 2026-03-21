# tests/unit/adaptive_learning/test_config.py
import pytest
from homie_core.config import AdaptiveLearningConfig, PreferenceLearningConfig, PerformanceLearningConfig, KnowledgeLearningConfig


class TestAdaptiveLearningConfig:
    def test_defaults(self):
        cfg = AdaptiveLearningConfig()
        assert cfg.enabled is True
        assert cfg.feedback_loops is True

    def test_preference_defaults(self):
        cfg = PreferenceLearningConfig()
        assert cfg.learning_rate_explicit == 0.3
        assert cfg.learning_rate_implicit == 0.05
        assert cfg.min_signals_for_confidence == 10

    def test_performance_defaults(self):
        cfg = PerformanceLearningConfig()
        assert cfg.cache_enabled is True
        assert cfg.cache_max_entries == 500
        assert cfg.cache_similarity_threshold == 0.92

    def test_knowledge_defaults(self):
        cfg = KnowledgeLearningConfig()
        assert cfg.conversation_mining is True
        assert cfg.project_tracking is True
        assert cfg.behavioral_profiling is True
