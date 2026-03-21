"""Context window optimizer — learns which context sources are useful per query type."""

import threading
from typing import Optional

from ..storage import LearningStorage


class ContextOptimizer:
    """Learns relevance of context sources per query type."""

    def __init__(self, storage: LearningStorage, learning_rate: float = 0.1) -> None:
        self._storage = storage
        self._lr = learning_rate
        self._lock = threading.Lock()
        # {(query_type, source): (relevance_score, sample_count)}
        self._scores: dict[tuple[str, str], tuple[float, int]] = {}

    def get_relevance(self, query_type: str, context_source: str) -> float:
        """Get learned relevance score for a context source."""
        with self._lock:
            entry = self._scores.get((query_type, context_source))
            return entry[0] if entry else 0.5

    def record_usage(self, query_type: str, context_source: str, was_referenced: bool) -> None:
        """Record whether a context source was useful for a query type."""
        target = 1.0 if was_referenced else 0.0
        with self._lock:
            current, count = self._scores.get((query_type, context_source), (0.5, 0))
            new_score = self._lr * target + (1 - self._lr) * current
            new_count = count + 1
            self._scores[(query_type, context_source)] = (new_score, new_count)

    def rank_sources(self, query_type: str, available_sources: list[str]) -> list[str]:
        """Rank context sources by learned relevance for a query type."""
        scored = [(s, self.get_relevance(query_type, s)) for s in available_sources]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored]
