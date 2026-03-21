"""Hybrid search — BM25 + vector similarity with Reciprocal Rank Fusion.

Two retrieval paths:
1. BM25 (keyword) — fast, handles exact matches, acronyms, variable names
2. Vector (semantic) — handles paraphrases, conceptual similarity

Results are fused with Reciprocal Rank Fusion (RRF) to combine the strengths
of both approaches.

The BM25 implementation is pure Python — no external dependencies.
Vector search delegates to VectorStore (ChromaDB).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# BM25 index (pure Python, in-memory)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer with basic normalization."""
    return re.findall(r"[a-zA-Z_]\w*", text.lower())


@dataclass
class BM25Index:
    """Okapi BM25 ranking over a corpus of documents.

    Parameters:
        k1: Term frequency saturation. Higher = more weight to repeated terms. (1.2-2.0 typical)
        b: Length normalization. 0 = no normalization, 1 = full. (0.75 typical)
    """
    k1: float = 1.5
    b: float = 0.75

    # Internal state
    _docs: list[list[str]] = field(default_factory=list)
    _doc_ids: list[str] = field(default_factory=list)
    _doc_metadata: list[dict] = field(default_factory=list)
    _doc_len: list[int] = field(default_factory=list)
    _avgdl: float = 0.0
    _df: dict[str, int] = field(default_factory=dict)  # document frequency
    _n_docs: int = 0

    def add(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        """Add a document to the index."""
        tokens = _tokenize(text)
        self._docs.append(tokens)
        self._doc_ids.append(doc_id)
        self._doc_metadata.append(metadata or {})
        self._doc_len.append(len(tokens))
        self._n_docs += 1

        # Update document frequency
        for word in set(tokens):
            self._df[word] = self._df.get(word, 0) + 1

        # Update average document length
        self._avgdl = sum(self._doc_len) / self._n_docs if self._n_docs else 0.0

    def remove(self, doc_id: str) -> None:
        """Remove a document from the index."""
        try:
            idx = self._doc_ids.index(doc_id)
        except ValueError:
            return

        tokens = self._docs[idx]
        for word in set(tokens):
            self._df[word] = max(0, self._df.get(word, 1) - 1)

        self._docs.pop(idx)
        self._doc_ids.pop(idx)
        self._doc_metadata.pop(idx)
        self._doc_len.pop(idx)
        self._n_docs -= 1
        self._avgdl = sum(self._doc_len) / self._n_docs if self._n_docs else 0.0

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search the index with BM25 scoring.

        Returns list of {id, score, metadata} sorted by descending score.
        """
        query_tokens = _tokenize(query)
        if not query_tokens or self._n_docs == 0:
            return []

        scores = [0.0] * self._n_docs

        for term in query_tokens:
            if term not in self._df:
                continue

            # IDF component
            df = self._df[term]
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for i, doc_tokens in enumerate(self._docs):
                # Term frequency in this document
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue

                # BM25 score component
                dl = self._doc_len[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * numerator / denominator

        # Build results sorted by score
        results = []
        for i, score in enumerate(scores):
            if score > 0:
                results.append({
                    "id": self._doc_ids[i],
                    "score": score,
                    "metadata": self._doc_metadata[i],
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @property
    def size(self) -> int:
        return self._n_docs


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *result_lists: list[dict[str, Any]],
    k: int = 60,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists using RRF.

    Each result_list is a list of dicts with at least an "id" key.
    RRF score = sum(1 / (k + rank)) across all lists.

    Args:
        result_lists: Multiple ranked result lists to fuse
        k: RRF constant (higher = more weight to lower-ranked results). Default 60.
        top_n: Number of results to return

    Returns:
        Fused results sorted by RRF score, with original metadata preserved.
    """
    rrf_scores: dict[str, float] = {}
    id_to_entry: dict[str, dict] = {}

    for results in result_lists:
        for rank, entry in enumerate(results):
            doc_id = entry["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            # Keep the entry with highest original score for metadata
            if doc_id not in id_to_entry:
                id_to_entry[doc_id] = entry

    # Sort by fused score
    fused = []
    for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(id_to_entry[doc_id])
        entry["rrf_score"] = score
        fused.append(entry)

    return fused[:top_n]


# ---------------------------------------------------------------------------
# Hybrid Search orchestrator
# ---------------------------------------------------------------------------

class HybridSearch:
    """Combines BM25 keyword search with vector semantic search.

    Usage:
        search = HybridSearch(vector_store)
        search.index_chunk(chunk_id, text, metadata)
        results = search.search("how does auth work?", top_k=5)
    """

    def __init__(
        self,
        vector_store=None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        reranker=None,
    ):
        self._vector_store = vector_store
        self._bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self._texts: dict[str, str] = {}  # id -> text for returning full chunks
        self._reranker = reranker

    def index_chunk(self, chunk_id: str, text: str, metadata: Optional[dict] = None) -> None:
        """Index a chunk in both BM25 and vector stores."""
        meta = metadata or {}
        self._bm25.add(chunk_id, text, meta)
        self._texts[chunk_id] = text

        if self._vector_store:
            # Convert metadata values to strings for ChromaDB
            str_meta = {k: str(v) for k, v in meta.items()}
            self._vector_store.add_file_chunk(chunk_id, text, str_meta)

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a chunk from both indexes."""
        self._bm25.remove(chunk_id)
        self._texts.pop(chunk_id, None)

        if self._vector_store:
            try:
                self._vector_store.delete_file_chunks([chunk_id])
            except Exception:
                pass

    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining BM25 and vector similarity.

        Returns results with text, metadata, and fused scores.
        """
        # BM25 search
        bm25_results = self._bm25.search(query, top_k=top_k * 2)

        # Vector search
        vector_results = []
        if self._vector_store:
            try:
                raw = self._vector_store.query_files(query, n=top_k * 2)
                vector_results = [
                    {"id": r["id"], "score": 1.0 / (1.0 + r.get("distance", 1.0)), "metadata": r.get("metadata", {})}
                    for r in raw
                ]
            except Exception:
                pass

        # Fuse results
        if bm25_results and vector_results:
            fused = reciprocal_rank_fusion(bm25_results, vector_results, top_n=top_k * 5)
        elif bm25_results:
            fused = bm25_results[: top_k * 5]
        elif vector_results:
            fused = vector_results[: top_k * 5]
        else:
            return []

        # Attach full text to results
        for entry in fused:
            entry["text"] = self._texts.get(entry["id"], "")
            # Also try from vector results
            if not entry["text"]:
                for vr in (vector_results or []):
                    if vr["id"] == entry["id"]:
                        entry["text"] = vr.get("text", "")
                        break

        # Optional reranking: rerank top-50 fused results, then return top-K
        if self._reranker is not None and self._reranker.available:
            candidates = fused[:50]
            texts = [entry.get("text", "") for entry in candidates]
            ranked = self._reranker.rerank(query, texts, top_k=top_k)
            return [candidates[orig_idx] for orig_idx, _score in ranked]

        return fused[:top_k]

    @property
    def size(self) -> int:
        return self._bm25.size
