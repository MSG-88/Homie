from __future__ import annotations

from collections import deque
from typing import Any, Callable

from homie_core.neural.utils import cosine_similarity, weighted_average, top_k_similar


class SemanticContextEngine:
    """Tracks semantic context using embedding vectors.

    Maintains a rolling context vector (exponentially-weighted average
    of recent activity embeddings) and detects semantic context shifts.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]],
                 embed_dim: int = 384,
                 window_size: int = 20,
                 decay: float = 0.85,
                 shift_threshold: float = 0.5):
        self._embed_fn = embed_fn
        self._embed_dim = embed_dim
        self._window_size = window_size
        self._decay = decay
        self._shift_threshold = shift_threshold
        self._context_vector: list[float] = [0.0] * embed_dim
        self._prev_context_vector: list[float] = [0.0] * embed_dim
        self._recent_embeddings: deque[list[float]] = deque(maxlen=window_size)
        self._observation_count = 0

    def update(self, process: str, title: str) -> None:
        """Process a new activity observation."""
        text = f"{process} {title}"
        embedding = self._embed_fn(text)

        self._prev_context_vector = list(self._context_vector)
        self._recent_embeddings.append(embedding)
        self._observation_count += 1

        # Exponentially-weighted average
        weights = []
        for i in range(len(self._recent_embeddings)):
            age = len(self._recent_embeddings) - 1 - i
            weights.append(self._decay ** age)

        self._context_vector = weighted_average(
            list(self._recent_embeddings), weights,
        )

    def get_context_vector(self) -> list[float]:
        """Return the current context vector."""
        return list(self._context_vector)

    def detect_context_shift(self) -> bool:
        """Check if context has shifted significantly."""
        if self._observation_count < 2:
            return False
        sim = cosine_similarity(self._context_vector, self._prev_context_vector)
        return sim < self._shift_threshold

    def find_relevant_memories(self, memories: list[dict],
                               top_k: int = 5) -> list[dict]:
        """Find memories most relevant to current context."""
        if not memories or self._observation_count == 0:
            return []

        embeddings = []
        valid_memories = []
        for m in memories:
            emb = m.get("embedding")
            if emb:
                embeddings.append(emb)
                valid_memories.append(m)

        if not embeddings:
            return []

        results = top_k_similar(self._context_vector, embeddings, top_k)
        return [valid_memories[idx] for idx, _ in results]

    def get_activity_summary(self) -> dict:
        """Return summary of current context state."""
        return {
            "observations": self._observation_count,
            "window_size": len(self._recent_embeddings),
            "has_context": self._observation_count > 0,
        }
