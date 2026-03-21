"""Cross-encoder reranker for RAG retrieval results.

Reranks retrieved documents using a cross-encoder model.
Falls back to identity (no reranking) if sentence-transformers is not installed.
"""
from __future__ import annotations

from typing import Optional


class CrossEncoderReranker:
    """Reranks retrieved documents using a cross-encoder model.

    Falls back to identity (no reranking) if sentence-transformers not installed.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model: Optional[object] = None
        self._model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
        except (ImportError, Exception):
            pass  # graceful degradation

    @property
    def available(self) -> bool:
        """True if the cross-encoder model was loaded successfully."""
        return self._model is not None

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Returns list of (original_index, score) sorted by score descending.
        If model not available, returns documents in original order with dummy scores.
        """
        if not documents:
            return []

        if self._model is None:
            # Identity fallback — preserve original order with decreasing dummy scores
            return [(i, 1.0 - i * 0.01) for i in range(min(len(documents), top_k))]

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)  # type: ignore[attr-defined]
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in indexed_scores[:top_k]]
