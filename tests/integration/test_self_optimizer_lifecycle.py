"""Integration test: self-optimizer lifecycle."""
import pytest
from homie_core.adaptive_learning.performance.self_optimizer.coordinator import SelfOptimizer
from homie_core.adaptive_learning.storage import LearningStorage


class TestSelfOptimizerLifecycle:
    def test_full_optimization_cycle(self, tmp_path):
        """Full flow: optimize -> execute -> record -> profile improves."""
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="test_hw")

        # First query — uses defaults
        params = opt.optimize_query("moderate", query_hint="coding")
        assert params["effective_complexity"] == "moderate"
        assert "temperature" in params

        # Record results over multiple queries
        for i in range(10):
            opt.record_result("coding", "moderate", temperature=0.3, max_tokens=512, response_tokens=250, latency_ms=200)

        # Profile should now have learned
        profile = opt.profiler.get_profile("coding")
        assert profile is not None
        assert profile.sample_count == 10
        assert profile.avg_response_tokens > 0

    def test_pipeline_gate_promotes_on_clarifications(self, tmp_path):
        """Gate promotes tier after repeated clarification requests."""
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="hw", promotion_threshold=2)

        # Trivial queries cause clarifications
        opt.record_clarification("trivial")
        opt.record_clarification("trivial")

        # Next trivial query should be promoted
        params = opt.optimize_query("trivial")
        assert params["effective_complexity"] == "simple"

    def test_prompt_compression(self, tmp_path):
        """Prompt optimizer compresses based on complexity."""
        storage = LearningStorage(db_path=tmp_path / "learn.db")
        storage.initialize()
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="hw")

        long_prompt = "System prompt.\n" + "Fact: " * 500
        opt.prompt_optimizer.set_complexity("trivial")
        compressed = opt.prompt_optimizer.modify_prompt(long_prompt)
        assert len(compressed) < len(long_prompt)
