from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from homie_core.behavioral.base import BaseObserver


class BrowsingObserver(BaseObserver):
    def __init__(self):
        super().__init__(name="browsing")
        self._site_time: dict[str, float] = defaultdict(float)
        self._search_queries: list[str] = []
        self._topic_counts: dict[str, int] = defaultdict(int)

    def tick(self) -> dict[str, Any]:
        return {}

    def observe_url(self, url: str, title: str, duration_seconds: float = 0) -> None:
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc
        except Exception:
            domain = url
        self._site_time[domain] += duration_seconds
        self.record({"type": "page_visit", "url": url, "title": title, "domain": domain})

    def get_profile_updates(self) -> dict[str, Any]:
        top_sites = sorted(self._site_time.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "top_sites": [s[0] for s in top_sites],
            "search_queries_count": len(self._search_queries),
        }
