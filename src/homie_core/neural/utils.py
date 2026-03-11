from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def weighted_average(vectors: list[list[float]],
                     weights: list[float]) -> list[float]:
    """Compute weighted average of vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    total_weight = sum(weights)
    if total_weight == 0.0:
        return [0.0] * dim
    result = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            result[i] += vec[i] * w
    return [x / total_weight for x in result]


def top_k_similar(query: list[float],
                  candidates: list[list[float]],
                  k: int) -> list[tuple[int, float]]:
    """Return top-k most similar candidates by cosine similarity.

    Returns list of (index, similarity) tuples, sorted descending.
    """
    scored = []
    for i, cand in enumerate(candidates):
        sim = cosine_similarity(query, cand)
        scored.append((i, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
