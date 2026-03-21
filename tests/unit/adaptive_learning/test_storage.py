import json
import pytest
from homie_core.adaptive_learning.storage import LearningStorage
from homie_core.adaptive_learning.observation.signals import (
    LearningSignal,
    SignalType,
    SignalCategory,
)


class TestLearningStorage:
    def test_initialize_creates_tables(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        tables = store.list_tables()
        assert "learning_signals" in tables
        assert "preference_profiles" in tables
        assert "response_cache" in tables
        assert "context_relevance" in tables
        assert "resource_patterns" in tables
        assert "project_knowledge" in tables
        assert "behavioral_patterns" in tables
        assert "decisions_log" in tables
        assert "customization_history" in tables

    def test_write_and_query_signal(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        sig = LearningSignal(
            signal_type=SignalType.EXPLICIT,
            category=SignalCategory.PREFERENCE,
            source="test",
            data={"dim": "verbosity"},
            context={"topic": "coding"},
        )
        store.write_signal(sig)
        results = store.query_signals(category="preference", limit=10)
        assert len(results) == 1
        assert results[0]["source"] == "test"

    def test_write_and_get_preference(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_preference("global", "default", {"verbosity": 0.3, "formality": 0.5})
        pref = store.get_preference("global", "default")
        assert pref is not None
        assert pref["verbosity"] == 0.3

    def test_preference_upsert(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.save_preference("global", "default", {"verbosity": 0.5})
        store.save_preference("global", "default", {"verbosity": 0.3})
        pref = store.get_preference("global", "default")
        assert pref["verbosity"] == 0.3

    def test_write_and_query_decision(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.write_decision("Use async over threading", "coding", {"project": "homie"})
        decisions = store.query_decisions(domain="coding")
        assert len(decisions) == 1
        assert "async" in decisions[0]["decision"]

    def test_close(self, tmp_path):
        store = LearningStorage(db_path=tmp_path / "learn.db")
        store.initialize()
        store.close()
        store.close()  # double close should not raise
