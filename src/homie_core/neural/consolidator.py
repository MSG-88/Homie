from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from homie_core.neural.utils import cosine_similarity, top_k_similar


@dataclass
class Pattern:
    """A recurring pattern found in episodes."""
    description: str
    episode_indices: list[int] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)
    frequency: int = 0


@dataclass
class ConsolidationResult:
    """Result of neural memory consolidation."""
    relevant: list[dict]
    clusters: list[Pattern]


class NeuralConsolidator:
    """Embedding-powered memory consolidation.

    Clusters episodic memories by semantic similarity, finds recurring
    patterns, and computes relevance-weighted decay scores.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]],
                 similarity_threshold: float = 0.7):
        self._embed_fn = embed_fn
        self._similarity_threshold = similarity_threshold

    def compute_relevance(self, memory: dict,
                          context: list[float]) -> float:
        """Compute relevance of a memory to the current context.

        Returns float 0-1. Higher = more relevant.
        """
        embedding = memory.get("embedding")
        if not embedding or not context:
            return 0.0
        sim = cosine_similarity(embedding, context)
        # Normalize from [-1, 1] to [0, 1]
        return max(0.0, min(1.0, (sim + 1) / 2))

    def find_patterns(self, episodes: list[dict]) -> list[Pattern]:
        """Find recurring patterns by clustering similar episodes.

        Uses simple greedy single-linkage clustering.
        """
        if not episodes:
            return []

        embeddings = []
        valid_indices = []
        for i, ep in enumerate(episodes):
            emb = ep.get("embedding")
            if emb:
                embeddings.append(emb)
                valid_indices.append(i)

        if not embeddings:
            return []

        # Greedy clustering
        assigned = [False] * len(embeddings)
        patterns = []

        for i in range(len(embeddings)):
            if assigned[i]:
                continue

            cluster_indices = [valid_indices[i]]
            cluster_embeddings = [embeddings[i]]
            assigned[i] = True

            for j in range(i + 1, len(embeddings)):
                if assigned[j]:
                    continue
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._similarity_threshold:
                    cluster_indices.append(valid_indices[j])
                    cluster_embeddings.append(embeddings[j])
                    assigned[j] = True

            if len(cluster_indices) >= 2:
                # Compute centroid
                dim = len(cluster_embeddings[0])
                centroid = [0.0] * dim
                for emb in cluster_embeddings:
                    for d in range(dim):
                        centroid[d] += emb[d]
                centroid = [c / len(cluster_embeddings) for c in centroid]

                summaries = [episodes[idx].get("summary", "")
                             for idx in cluster_indices]
                desc = f"Recurring pattern ({len(cluster_indices)} episodes): " + \
                       ", ".join(summaries[:3])

                patterns.append(Pattern(
                    description=desc,
                    episode_indices=cluster_indices,
                    centroid=centroid,
                    frequency=len(cluster_indices),
                ))

        return patterns

    def consolidate(self, episodes: list[dict],
                    current_context: list[float]) -> dict:
        """Run full consolidation: relevance scoring + pattern finding.

        Returns dict with 'relevant' (sorted episodes) and 'clusters' (patterns).
        """
        if not episodes:
            return {"relevant": [], "clusters": []}

        # Score relevance to current context
        scored = []
        for ep in episodes:
            score = self.compute_relevance(ep, current_context)
            scored.append({**ep, "_relevance": score})

        scored.sort(key=lambda x: x["_relevance"], reverse=True)

        # Find patterns
        clusters = self.find_patterns(episodes)

        return {
            "relevant": scored,
            "clusters": clusters,
        }
