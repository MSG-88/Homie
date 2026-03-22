import time
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.knowledge.graph.temporal import TemporalManager


class TestTemporalManager:
    def test_supersede_relationship(self):
        graph = MagicMock()
        old_rel = MagicMock(valid_until=None, subject_id="e1", relation="works_at", object_id="e2")
        graph.find_current_relationships.return_value = [old_rel]
        tm = TemporalManager(graph_store=graph)
        tm.supersede("e1", "works_at", "e3")
        assert old_rel.valid_until is not None  # was set
        graph.update_relationship.assert_called()

    def test_no_supersede_when_no_existing(self):
        graph = MagicMock()
        graph.find_current_relationships.return_value = []
        tm = TemporalManager(graph_store=graph)
        tm.supersede("e1", "works_at", "e3")
        graph.update_relationship.assert_not_called()

    def test_confidence_decay(self):
        tm = TemporalManager(graph_store=MagicMock(), decay_rate=0.99)
        base_confidence = 0.8
        age_days = 70
        decayed = tm.apply_decay(base_confidence, age_days)
        assert decayed < base_confidence
        assert decayed > 0

    def test_no_decay_for_fresh_facts(self):
        tm = TemporalManager(graph_store=MagicMock(), decay_rate=0.99)
        assert tm.apply_decay(0.8, age_days=0) == pytest.approx(0.8)

    def test_query_at_time(self):
        graph = MagicMock()
        tm = TemporalManager(graph_store=graph)
        tm.query_at_time("e1", "works_at", timestamp=time.time())
        graph.find_relationships_at_time.assert_called()
