"""TrainingTrigger — Automated fine-tuning decision based on accumulated feedback."""
from __future__ import annotations

from homie_core.mesh.feedback_store import FeedbackStore


class TrainingTrigger:
    """Decide whether enough feedback signals have accumulated to start fine-tuning.

    Trigger conditions (OR logic):
    - ``new >= min_signals``: enough new signals since last trigger.
    - ``corrections >= min_corrections``: enough correction signals regardless of total.
    """

    def __init__(
        self,
        feedback_store: FeedbackStore,
        min_signals: int = 500,
        min_corrections: int = 100,
    ) -> None:
        self._store = feedback_store
        self._min_signals = min_signals
        self._min_corrections = min_corrections
        self._last_triggered_count: int = 0

    def should_trigger(self) -> bool:
        """Return True if training should start now."""
        total = self._store.total_count()
        new = total - self._last_triggered_count
        if new <= 0:
            return False
        if new >= self._min_signals:
            return True
        return self._store.count_by_type().get("corrected", 0) >= self._min_corrections

    def mark_triggered(self) -> None:
        """Record that training was launched; resets the new-signal counter."""
        self._last_triggered_count = self._store.total_count()

    def get_summary(self) -> dict:
        """Return a snapshot dict describing the current state."""
        total = self._store.total_count()
        pairs = self._store.get_training_pairs()
        return {
            "total_signals": total,
            "by_type": self._store.count_by_type(),
            "sft_pairs": sum(1 for p in pairs if p["type"] == "sft"),
            "dpo_pairs": sum(1 for p in pairs if p["type"] == "dpo"),
            "new_since_last": total - self._last_triggered_count,
            "ready": self.should_trigger(),
        }
