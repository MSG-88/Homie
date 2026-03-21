"""Integration test: full adaptive learning lifecycle."""
import pytest
from homie_core.adaptive_learning.learner import AdaptiveLearner


class TestAdaptiveLearningLifecycle:
    def test_full_turn_lifecycle(self, tmp_path):
        """Full flow: signal -> preference update -> prompt layer."""
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")

        # Process multiple turns with explicit preferences
        for _ in range(15):
            learner.process_turn("Please be more concise", "Ok, I'll be brief.", state={"topic": "general"})

        # Preference should have shifted
        profile = learner.preference_engine.get_active_profile()
        assert profile.verbosity < 0.4  # should have decreased

        # Prompt layer should reflect preference
        prompt = learner.get_prompt_layer()
        assert "concise" in prompt.lower() or "brief" in prompt.lower() or "short" in prompt.lower()

        learner.stop()

    def test_knowledge_extraction(self, tmp_path):
        """Facts are extracted from conversation."""
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")
        learner.process_turn("I work at Google as a data scientist", "That's great!", state={})

        # Check storage has the fact
        decisions = learner._storage.query_decisions()
        assert len(decisions) >= 1

        learner.stop()

    def test_cache_lifecycle(self, tmp_path):
        """Response cache stores and retrieves."""
        learner = AdaptiveLearner(db_path=tmp_path / "learn.db")
        learner.performance_optimizer.cache_response("What is Python?", "A programming language.")
        result = learner.get_cached_response("What is Python?")
        assert result == "A programming language."

        learner.stop()
