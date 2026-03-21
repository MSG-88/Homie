"""Tests for CrossEncoderReranker — TDD for SP3 Advanced Retrieval."""
from __future__ import annotations

import pytest
from homie_core.rag.reranker import CrossEncoderReranker


class TestCrossEncoderRerankerFallback:
    """Tests that must pass even without sentence-transformers installed."""

    def test_empty_documents_returns_empty(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"
        result = reranker.rerank("query", [])
        assert result == []

    def test_fallback_preserves_original_order(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"
        docs = ["first", "second", "third"]
        result = reranker.rerank("query", docs, top_k=3)
        indices = [idx for idx, score in result]
        assert indices == [0, 1, 2]

    def test_fallback_scores_are_decreasing(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"
        docs = ["a", "b", "c", "d"]
        result = reranker.rerank("query", docs, top_k=4)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_fallback_respects_top_k(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"
        docs = ["a", "b", "c", "d", "e"]
        result = reranker.rerank("query", docs, top_k=3)
        assert len(result) == 3

    def test_fallback_top_k_larger_than_docs(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"
        docs = ["a", "b"]
        result = reranker.rerank("query", docs, top_k=10)
        assert len(result) == 2

    def test_available_false_without_model(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"
        assert reranker.available is False

    def test_available_true_with_model(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = object()  # sentinel non-None model
        reranker._model_name = "test-model"
        assert reranker.available is True

    def test_default_constructor_does_not_raise(self):
        """Constructor must not raise regardless of whether sentence-transformers is installed."""
        reranker = CrossEncoderReranker()
        # Either available (installed) or not available (graceful fallback)
        assert isinstance(reranker.available, bool)

    def test_rerank_returns_list_of_tuples(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"
        docs = ["doc1", "doc2"]
        result = reranker.rerank("query", docs)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_single_document_fallback(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"
        result = reranker.rerank("query", ["only doc"])
        assert len(result) == 1
        assert result[0][0] == 0


class TestCrossEncoderRerankerWithModel:
    """Tests using real model — skipped if sentence-transformers not installed."""

    @pytest.fixture
    def reranker_with_model(self):
        CrossEncoder = pytest.importorskip(
            "sentence_transformers", reason="sentence-transformers not installed"
        )
        r = CrossEncoderReranker()
        if not r.available:
            pytest.skip("CrossEncoder model failed to load")
        return r

    def test_reranks_by_relevance(self, reranker_with_model):
        query = "python authentication"
        docs = [
            "unrelated document about cooking recipes",
            "python authentication and login security",
            "machine learning deep neural networks",
        ]
        result = reranker_with_model.rerank(query, docs, top_k=3)
        assert len(result) == 3
        # Most relevant doc (index 1) should rank highest
        top_idx = result[0][0]
        assert top_idx == 1

    def test_scores_are_floats(self, reranker_with_model):
        result = reranker_with_model.rerank("test", ["doc a", "doc b"])
        for idx, score in result:
            assert isinstance(score, float)
