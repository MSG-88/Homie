# tests/unit/self_optimizer/test_coordinator.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.performance.self_optimizer.coordinator import SelfOptimizer


class TestSelfOptimizer:
    def test_initializes_all_components(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc123")
        assert opt.prompt_optimizer is not None
        assert opt.model_tuner is not None
        assert opt.pipeline_gate is not None
        assert opt.profiler is not None

    def test_optimize_query_returns_params(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc")
        result = opt.optimize_query(complexity="moderate", query_hint="coding")
        assert "temperature" in result
        assert "max_tokens" in result
        assert "effective_complexity" in result

    def test_trivial_query_gets_fast_params(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc")
        result = opt.optimize_query(complexity="trivial")
        assert result["max_tokens"] <= 256
        assert result["effective_complexity"] == "trivial"

    def test_record_result_updates_profiler(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc")
        opt.record_result(
            query_type="coding",
            complexity="moderate",
            temperature=0.5,
            max_tokens=512,
            response_tokens=200,
            latency_ms=300,
        )
        storage.save_optimization_profile.assert_called()

    def test_record_clarification_feeds_gate(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc", promotion_threshold=1)
        opt.record_clarification("trivial")
        result = opt.optimize_query(complexity="trivial")
        assert result["effective_complexity"] == "simple"  # promoted

    def test_get_stats(self):
        storage = MagicMock()
        storage.get_optimization_profile.return_value = None
        opt = SelfOptimizer(storage=storage, hardware_fingerprint="abc")
        opt.optimize_query(complexity="moderate")
        stats = opt.get_stats()
        assert "gate" in stats
