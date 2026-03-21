# tests/unit/adaptive_learning/test_learner.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.learner import AdaptiveLearner


class TestAdaptiveLearner:
    def test_initializes_all_engines(self, tmp_path):
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")
        assert learner.preference_engine is not None
        assert learner.performance_optimizer is not None
        assert learner.knowledge_builder is not None
        assert learner.observation_stream is not None

    def test_process_turn_feeds_all_engines(self, tmp_path):
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")
        learner.process_turn("I prefer bullet points", "Sure, I'll use bullets.", state={"topic": "general"})
        # Should have updated preference
        profile = learner.preference_engine.get_active_profile()
        assert profile.format_preference == "bullets"

    def test_get_prompt_layer(self, tmp_path):
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")
        # Feed enough signals
        for _ in range(15):
            learner.process_turn("be more concise", "Ok.", state={})
        prompt = learner.get_prompt_layer()
        assert isinstance(prompt, str)

    def test_start_and_stop(self, tmp_path):
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")
        learner.start()
        learner.stop()

    def test_cache_hit(self, tmp_path):
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")
        learner.performance_optimizer.cache_response("test query", "cached response")
        result = learner.get_cached_response("test query")
        assert result == "cached response"
