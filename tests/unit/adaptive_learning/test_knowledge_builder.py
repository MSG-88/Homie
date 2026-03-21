# tests/unit/adaptive_learning/test_knowledge_builder.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.knowledge.builder import KnowledgeBuilder
from homie_core.adaptive_learning.observation.signals import (
    LearningSignal,
    SignalType,
    SignalCategory,
)


class TestKnowledgeBuilder:
    def _make_builder(self, storage=None):
        storage = storage or MagicMock()
        return KnowledgeBuilder(storage=storage)

    def test_on_signal_behavioral(self):
        builder = self._make_builder()
        sig = LearningSignal(
            signal_type=SignalType.BEHAVIORAL,
            category=SignalCategory.CONTEXT,
            source="app_tracker",
            data={"hour": 9, "app": "VSCode"},
            context={},
        )
        builder.on_signal(sig)
        pred = builder.profiler.predict(hour=9, category="app")
        assert pred == "VSCode"

    def test_process_turn_extracts_facts(self):
        storage = MagicMock()
        builder = KnowledgeBuilder(storage=storage)
        facts = builder.process_turn("I work at Google", "Nice!")
        assert len(facts) >= 1

    def test_register_and_list_projects(self):
        builder = self._make_builder()
        builder.project_tracker.register_project("homie", "/path/to/homie")
        projects = builder.project_tracker.list_projects()
        assert "homie" in projects

    def test_get_work_hours(self):
        builder = self._make_builder()
        for _ in range(5):
            builder.profiler.record_observation(hour=10, category="activity", value="coding")
        hours = builder.get_work_hours()
        assert 10 in hours
