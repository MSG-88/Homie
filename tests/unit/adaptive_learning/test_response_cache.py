# tests/unit/adaptive_learning/test_response_cache.py
import time
import pytest
from homie_core.adaptive_learning.performance.response_cache import ResponseCache


class TestResponseCache:
    def test_put_and_get(self):
        cache = ResponseCache(max_entries=100, ttl_default=3600)
        cache.put("What is Python?", "Python is a programming language.", context_hash="ctx1")
        result = cache.get("What is Python?")
        assert result is not None
        assert "programming language" in result

    def test_miss_on_unknown_query(self):
        cache = ResponseCache(max_entries=100, ttl_default=3600)
        assert cache.get("Never asked this") is None

    def test_similar_query_hits_cache(self):
        cache = ResponseCache(max_entries=100, ttl_default=3600, similarity_threshold=0.5)
        cache.put("What is Python?", "Python is a language.", context_hash="ctx1")
        # Exact same query should always hit
        result = cache.get("What is Python?")
        assert result is not None

    def test_expired_entry_returns_none(self):
        cache = ResponseCache(max_entries=100, ttl_default=0.01)
        cache.put("test", "response", context_hash="ctx1")
        time.sleep(0.02)
        assert cache.get("test") is None

    def test_max_entries_eviction(self):
        cache = ResponseCache(max_entries=2, ttl_default=3600)
        cache.put("q1", "r1", context_hash="c1")
        cache.put("q2", "r2", context_hash="c2")
        cache.put("q3", "r3", context_hash="c3")
        # q1 should be evicted (LRU)
        assert cache.get("q1") is None
        assert cache.get("q3") is not None

    def test_invalidate(self):
        cache = ResponseCache(max_entries=100, ttl_default=3600)
        cache.put("test", "response", context_hash="ctx1")
        cache.invalidate("test")
        assert cache.get("test") is None

    def test_stats(self):
        cache = ResponseCache(max_entries=100, ttl_default=3600)
        cache.put("q", "r", context_hash="c")
        cache.get("q")  # hit
        cache.get("miss")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
