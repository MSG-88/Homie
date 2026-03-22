"""Temporal versioning — manages time ranges on relationships, confidence decay."""

import time
from typing import Any, Optional


class TemporalManager:
    """Manages temporal versioning and confidence decay for knowledge relationships."""

    def __init__(self, graph_store, decay_rate: float = 0.99) -> None:
        self._graph = graph_store
        self._decay_rate = decay_rate

    def supersede(self, subject_id: str, relation: str, new_object_id: str) -> None:
        """Supersede existing current relationships with a new one."""
        now = time.time()
        existing = self._graph.find_current_relationships(subject_id, relation)
        for rel in existing:
            if rel.object_id != new_object_id:
                rel.valid_until = now
                self._graph.update_relationship(rel)

    def apply_decay(self, base_confidence: float, age_days: float) -> float:
        """Apply time-based confidence decay."""
        return base_confidence * (self._decay_rate ** age_days)

    def query_at_time(
        self, subject_id: str, relation: str, timestamp: float
    ) -> list:
        """Query relationships that were valid at a specific timestamp."""
        return self._graph.find_relationships_at_time(subject_id, relation, timestamp)
