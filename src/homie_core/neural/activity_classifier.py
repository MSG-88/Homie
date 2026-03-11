from __future__ import annotations

import math
from typing import Callable, Optional

from homie_core.neural.utils import cosine_similarity


CATEGORIES = [
    "coding", "researching", "communicating", "writing",
    "designing", "browsing", "media", "system", "unknown",
]

# Prototype descriptions for zero-shot classification
_PROTOTYPE_DESCRIPTIONS = {
    "coding": "programming code editor IDE terminal compiler debug",
    "researching": "search documentation reference API reading learning",
    "communicating": "email chat message slack teams discord call meeting",
    "writing": "document writing text editor word notes markdown",
    "designing": "design figma photoshop graphics layout UI mockup",
    "browsing": "web browser internet social media news reddit",
    "media": "video music player spotify youtube streaming audio",
    "system": "settings control panel file manager task manager system",
    "unknown": "other miscellaneous general application",
}


class ActivityClassifier:
    """Classifies user activity into semantic categories using embeddings.

    Uses cosine similarity to category prototypes (zero-shot) initially,
    with optional online learning via a small feedforward layer.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]],
                 embed_dim: int = 384):
        self._embed_fn = embed_fn
        self._embed_dim = embed_dim
        self._prototypes: dict[str, list[float]] = {}
        # Simple linear layer for online learning: weights per category
        self._weights: dict[str, list[float]] = {}
        self._bias: dict[str, float] = {}
        self._n_samples = 0
        self._lr = 0.05

    def _init_prototypes(self) -> None:
        """Compute prototype embeddings for each category."""
        for cat, desc in _PROTOTYPE_DESCRIPTIONS.items():
            self._prototypes[cat] = self._embed_fn(desc)
            self._weights[cat] = [0.0] * self._embed_dim
            self._bias[cat] = 0.0

    def classify(self, process: str, title: str) -> dict[str, float]:
        """Classify activity into categories with confidence scores."""
        if not self._prototypes:
            self._init_prototypes()

        text = f"{process} {title}"
        embedding = self._embed_fn(text)

        # Cosine similarity to prototypes
        similarities = {}
        for cat, proto in self._prototypes.items():
            sim = cosine_similarity(embedding, proto)
            # Add learned adjustment
            learned = sum(
                w * x for w, x in zip(self._weights[cat], embedding)
            ) + self._bias[cat]
            similarities[cat] = sim + learned

        # Softmax normalization
        max_sim = max(similarities.values())
        exp_sims = {
            cat: math.exp(sim - max_sim)
            for cat, sim in similarities.items()
        }
        total = sum(exp_sims.values())
        return {cat: exp_sim / total for cat, exp_sim in exp_sims.items()}

    def get_top_activity(self, process: str, title: str) -> str:
        """Return the most likely activity category."""
        scores = self.classify(process, title)
        return max(scores, key=scores.get)

    def train_online(self, process: str, title: str, label: str) -> None:
        """Update classifier from a labeled observation via SGD."""
        if label not in CATEGORIES:
            return
        if not self._prototypes:
            self._init_prototypes()

        text = f"{process} {title}"
        embedding = self._embed_fn(text)
        scores = self.classify(process, title)

        # SGD update: increase score for correct label, decrease others
        for cat in CATEGORIES:
            target = 1.0 if cat == label else 0.0
            error = target - scores[cat]
            for i in range(len(embedding)):
                self._weights[cat][i] += self._lr * error * embedding[i]
            self._bias[cat] += self._lr * error

        self._n_samples += 1

    def serialize(self) -> dict:
        return {
            "prototypes": {k: list(v) for k, v in self._prototypes.items()},
            "weights": {k: list(v) for k, v in self._weights.items()},
            "bias": dict(self._bias),
            "n_samples": self._n_samples,
            "embed_dim": self._embed_dim,
        }

    @classmethod
    def deserialize(cls, data: dict,
                    embed_fn: Callable[[str], list[float]]) -> ActivityClassifier:
        obj = cls(embed_fn=embed_fn, embed_dim=data.get("embed_dim", 384))
        obj._prototypes = {k: list(v) for k, v in data["prototypes"].items()}
        obj._weights = {k: list(v) for k, v in data["weights"].items()}
        obj._bias = dict(data.get("bias", {}))
        obj._n_samples = data.get("n_samples", 0)
        return obj
