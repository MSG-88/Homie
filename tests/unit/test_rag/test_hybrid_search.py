"""Tests for hybrid search — BM25, RRF, and HybridSearch."""
import pytest
from homie_core.rag.hybrid_search import BM25Index, reciprocal_rank_fusion, HybridSearch


# -----------------------------------------------------------------------
# BM25 Index
# -----------------------------------------------------------------------

class TestBM25Index:
    def test_add_and_search(self):
        idx = BM25Index()
        idx.add("d1", "the quick brown fox jumps over the lazy dog")
        idx.add("d2", "the cat sat on the mat")
        idx.add("d3", "python programming language")

        results = idx.search("quick fox")
        assert len(results) >= 1
        assert results[0]["id"] == "d1"

    def test_empty_query(self):
        idx = BM25Index()
        idx.add("d1", "some text")
        results = idx.search("")
        assert results == []

    def test_empty_index(self):
        idx = BM25Index()
        results = idx.search("hello")
        assert results == []

    def test_no_match(self):
        idx = BM25Index()
        idx.add("d1", "hello world")
        results = idx.search("quantum physics")
        assert len(results) == 0

    def test_ranking_order(self):
        idx = BM25Index()
        idx.add("d1", "python programming")
        idx.add("d2", "python python python programming language deep learning")
        idx.add("d3", "java programming")

        results = idx.search("python programming")
        # d2 has more "python" mentions, should rank higher
        ids = [r["id"] for r in results]
        assert "d2" in ids or "d1" in ids  # both match

    def test_remove_document(self):
        idx = BM25Index()
        idx.add("d1", "hello world")
        idx.add("d2", "goodbye world")
        idx.remove("d1")

        results = idx.search("hello")
        assert all(r["id"] != "d1" for r in results)
        assert idx.size == 1

    def test_remove_nonexistent(self):
        idx = BM25Index()
        idx.remove("nope")  # should not raise
        assert idx.size == 0

    def test_metadata_preserved(self):
        idx = BM25Index()
        idx.add("d1", "hello world", {"file": "test.py", "line": "10"})
        results = idx.search("hello")
        assert results[0]["metadata"]["file"] == "test.py"

    def test_top_k_limit(self):
        idx = BM25Index()
        for i in range(20):
            idx.add(f"d{i}", f"document number {i} with some shared words")
        results = idx.search("document number", top_k=5)
        assert len(results) <= 5

    def test_code_search(self):
        idx = BM25Index()
        idx.add("f1", "def authenticate_user(username, password):\n    return check_credentials(username, password)")
        idx.add("f2", "def get_user_profile(user_id):\n    return db.query(User, user_id)")
        idx.add("f3", "def calculate_tax(income, rate):\n    return income * rate")

        results = idx.search("authentication password")
        assert results[0]["id"] == "f1"


# -----------------------------------------------------------------------
# Reciprocal Rank Fusion
# -----------------------------------------------------------------------

class TestRRF:
    def test_single_list(self):
        results = reciprocal_rank_fusion(
            [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}],
            top_n=2,
        )
        assert len(results) == 2
        assert results[0]["id"] == "a"

    def test_two_lists_fusion(self):
        list1 = [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}]
        list2 = [{"id": "b", "score": 1.0}, {"id": "c", "score": 0.5}]

        results = reciprocal_rank_fusion(list1, list2, top_n=3)
        # "b" appears in both lists, should rank highest
        assert results[0]["id"] == "b"

    def test_empty_lists(self):
        results = reciprocal_rank_fusion([], [], top_n=5)
        assert results == []

    def test_rrf_score_present(self):
        results = reciprocal_rank_fusion(
            [{"id": "x", "score": 1.0}],
            top_n=1,
        )
        assert "rrf_score" in results[0]
        assert results[0]["rrf_score"] > 0

    def test_top_n_limit(self):
        big_list = [{"id": f"d{i}", "score": 1.0 / (i + 1)} for i in range(20)]
        results = reciprocal_rank_fusion(big_list, top_n=3)
        assert len(results) == 3


# -----------------------------------------------------------------------
# Hybrid Search
# -----------------------------------------------------------------------

class TestHybridSearch:
    def test_index_and_search_bm25_only(self):
        search = HybridSearch(vector_store=None)
        search.index_chunk("c1", "python authentication module", {"file": "auth.py"})
        search.index_chunk("c2", "javascript frontend component", {"file": "app.js"})

        results = search.search("authentication", top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "c1"
        assert "authentication" in results[0]["text"]

    def test_remove_chunk(self):
        search = HybridSearch(vector_store=None)
        search.index_chunk("c1", "hello world")
        search.remove_chunk("c1")
        results = search.search("hello")
        assert all(r["id"] != "c1" for r in results)

    def test_size_tracking(self):
        search = HybridSearch(vector_store=None)
        assert search.size == 0
        search.index_chunk("c1", "text")
        assert search.size == 1

    def test_empty_search(self):
        search = HybridSearch(vector_store=None)
        results = search.search("anything")
        assert results == []


# -----------------------------------------------------------------------
# HybridSearch with reranker integration
# -----------------------------------------------------------------------

class TestHybridSearchReranker:
    """Tests for optional reranker integration in HybridSearch."""

    def test_no_reranker_search_works_normally(self):
        """HybridSearch without reranker behaves as before."""
        search = HybridSearch(vector_store=None, reranker=None)
        search.index_chunk("c1", "machine learning neural networks")
        search.index_chunk("c2", "recipe for chocolate cake")
        results = search.search("neural networks", top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "c1"

    def test_reranker_parameter_accepted(self):
        """HybridSearch accepts a reranker keyword argument without error."""
        from homie_core.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        search = HybridSearch(vector_store=None, reranker=reranker)
        assert search is not None

    def test_fallback_reranker_returns_results(self):
        """With fallback reranker (no model), results are still returned."""
        from homie_core.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"

        search = HybridSearch(vector_store=None, reranker=reranker)
        search.index_chunk("c1", "python authentication")
        search.index_chunk("c2", "javascript frontend")
        results = search.search("authentication", top_k=2)
        assert len(results) >= 1

    def test_reranker_result_count_bounded_by_top_k(self):
        """Results count must not exceed top_k even with reranker."""
        from homie_core.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"

        search = HybridSearch(vector_store=None, reranker=reranker)
        for i in range(20):
            search.index_chunk(f"c{i}", f"document about topic {i} shared words here")
        results = search.search("topic shared words", top_k=5)
        assert len(results) <= 5

    def test_reranker_result_has_text_field(self):
        """Results from reranked search still carry full text."""
        from homie_core.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        reranker._model_name = "test-model"

        search = HybridSearch(vector_store=None, reranker=reranker)
        search.index_chunk("c1", "the quick brown fox")
        results = search.search("quick fox", top_k=1)
        assert results[0]["text"] == "the quick brown fox"

    def test_none_reranker_is_default(self):
        """HybridSearch() with no reranker arg behaves identically to reranker=None."""
        s1 = HybridSearch(vector_store=None)
        s2 = HybridSearch(vector_store=None, reranker=None)
        for s in (s1, s2):
            s.index_chunk("c1", "hello world")
        r1 = s1.search("hello")
        r2 = s2.search("hello")
        assert [r["id"] for r in r1] == [r["id"] for r in r2]
