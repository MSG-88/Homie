from __future__ import annotations

from collections import deque
from typing import Optional

from homie_core.neural.utils import cosine_similarity, weighted_average


class IntentInferencer:
    """Predicts user intent from sequences of activity embeddings.

    Uses weighted k-NN over stored activity sequences. When the user's
    current activity sequence matches a previously seen pattern, predicts
    the next step.

    GPU/GRU upgrade path: replace _predict_knn with a trained GRU model
    when enough data is collected.
    """

    def __init__(self, embed_dim: int = 384,
                 sequence_length: int = 10,
                 min_sequences: int = 5):
        self._embed_dim = embed_dim
        self._sequence_length = sequence_length
        self._min_sequences = min_sequences
        self._sequence: deque[list[float]] = deque(maxlen=sequence_length)
        self._stored_sequences: list[list[list[float]]] = []

    def observe(self, activity_embedding: list[float]) -> None:
        """Add a new activity observation to the current sequence."""
        self._sequence.append(list(activity_embedding))

    def has_enough_data(self) -> bool:
        """Check if we have enough stored sequences for prediction."""
        return len(self._stored_sequences) >= self._min_sequences

    def predict_next(self) -> dict:
        """Predict the next activity based on current sequence.

        Returns dict with:
        - predicted_activity: embedding vector or None
        - confidence: float 0-1
        - estimated_completion: float 0-1 (how far through the pattern)
        """
        if not self._stored_sequences or len(self._sequence) < 2:
            return {
                "predicted_activity": None,
                "confidence": 0.0,
                "estimated_completion": 0.0,
            }

        current = list(self._sequence)
        best_sim = -1.0
        best_next = None
        best_position = 0.0

        for stored in self._stored_sequences:
            if len(stored) < 2:
                continue

            # Slide current sequence over stored to find best match
            for start in range(len(stored) - len(current)):
                window = stored[start:start + len(current)]
                if len(window) != len(current):
                    continue

                # Average similarity across the window
                sims = []
                for a, b in zip(current, window):
                    sims.append(cosine_similarity(a, b))
                avg_sim = sum(sims) / len(sims) if sims else 0.0

                if avg_sim > best_sim:
                    best_sim = avg_sim
                    next_idx = start + len(current)
                    if next_idx < len(stored):
                        best_next = stored[next_idx]
                    best_position = (start + len(current)) / len(stored)

        if best_next is None or best_sim < 0.3:
            return {
                "predicted_activity": None,
                "confidence": 0.0,
                "estimated_completion": 0.0,
            }

        return {
            "predicted_activity": best_next,
            "confidence": max(0.0, min(1.0, best_sim)),
            "estimated_completion": best_position,
        }

    def get_likely_needs(self) -> list[str]:
        """Return descriptions of likely upcoming information needs.

        Placeholder — returns empty until we have activity-to-need mapping.
        """
        return []

    def train_from_sequence(self, sequence: list[list[float]]) -> None:
        """Store a completed activity sequence for future matching."""
        if len(sequence) >= 2:
            self._stored_sequences.append([list(v) for v in sequence])

    def complete_current_sequence(self) -> None:
        """Mark current sequence as complete and store for training."""
        if len(self._sequence) >= 2:
            self._stored_sequences.append(list(self._sequence))
        self._sequence.clear()

    def serialize(self) -> dict:
        return {
            "embed_dim": self._embed_dim,
            "sequence_length": self._sequence_length,
            "min_sequences": self._min_sequences,
            "sequence": [list(v) for v in self._sequence],
            "sequences": [[list(v) for v in seq]
                          for seq in self._stored_sequences],
        }

    @classmethod
    def deserialize(cls, data: dict) -> IntentInferencer:
        obj = cls(
            embed_dim=data.get("embed_dim", 384),
            sequence_length=data.get("sequence_length", 10),
            min_sequences=data.get("min_sequences", 5),
        )
        for v in data.get("sequence", []):
            obj._sequence.append(list(v))
        obj._stored_sequences = [
            [list(v) for v in seq]
            for seq in data.get("sequences", [])
        ]
        return obj
