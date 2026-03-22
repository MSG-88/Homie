# tests/unit/self_optimizer/test_storage_integration.py
import pytest
from homie_core.adaptive_learning.storage import LearningStorage


class TestOptimizationProfileStorage:
    def test_save_and_get_profile(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        data = {"query_type": "coding", "hardware_fingerprint": "abc", "temperature": 0.4, "max_tokens": 800}
        store.save_optimization_profile("coding", "abc", data)
        result = store.get_optimization_profile("coding", "abc")
        assert result is not None
        assert result["temperature"] == 0.4

    def test_upsert_profile(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_optimization_profile("coding", "abc", {"temperature": 0.5})
        store.save_optimization_profile("coding", "abc", {"temperature": 0.3})
        result = store.get_optimization_profile("coding", "abc")
        assert result["temperature"] == 0.3

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        assert store.get_optimization_profile("unknown", "xyz") is None
