"""Semantic response cache with LRU eviction and TTL."""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    query: str
    query_hash: str
    response: str
    context_hash: str
    ttl: float
    created_at: float
    hit_count: int = 0
    last_hit: float = 0.0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class ResponseCache:
    """In-memory response cache with hash-based lookup, LRU eviction, and TTL."""

    def __init__(
        self,
        max_entries: int = 500,
        ttl_default: float = 86400.0,
        similarity_threshold: float = 0.92,
    ) -> None:
        self._max_entries = max_entries
        self._ttl_default = ttl_default
        self._similarity_threshold = similarity_threshold
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _hash_query(self, query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def put(
        self,
        query: str,
        response: str,
        context_hash: str,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a query-response pair."""
        qhash = self._hash_query(query)
        entry = CacheEntry(
            query=query,
            query_hash=qhash,
            response=response,
            context_hash=context_hash,
            ttl=ttl or self._ttl_default,
            created_at=time.time(),
        )
        with self._lock:
            if qhash in self._entries:
                del self._entries[qhash]
            self._entries[qhash] = entry
            # Evict oldest if over capacity
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def get(self, query: str, context_hash: Optional[str] = None) -> Optional[str]:
        """Look up a cached response. Returns None on miss."""
        qhash = self._hash_query(query)
        with self._lock:
            entry = self._entries.get(qhash)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._entries[qhash]
                self._misses += 1
                return None

            if context_hash and entry.context_hash != context_hash:
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._entries.move_to_end(qhash)
            entry.hit_count += 1
            entry.last_hit = time.time()
            self._hits += 1
            return entry.response

    def invalidate(self, query: str) -> None:
        """Remove a cached entry."""
        qhash = self._hash_query(query)
        with self._lock:
            self._entries.pop(qhash, None)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
            }
