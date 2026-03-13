from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from homie_core.utils import utc_now


class AppTracker:
    def __init__(self):
        import warnings
        warnings.warn(
            "AppTracker is deprecated. Use homie_core.screen_reader.window_tracker.WindowTracker instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._usage: dict[str, float] = defaultdict(float)  # app -> total seconds
        self._switches: list[dict] = []
        self._current_app: str | None = None
        self._current_start: datetime | None = None

    def track(self, app_name: str) -> None:
        now = utc_now()
        if self._current_app and self._current_app != app_name:
            elapsed = (now - self._current_start).total_seconds() if self._current_start else 0
            self._usage[self._current_app] += elapsed
            self._switches.append({
                "from": self._current_app,
                "to": app_name,
                "timestamp": now.isoformat(),
            })
        if self._current_app != app_name:
            self._current_app = app_name
            self._current_start = now

    def get_usage(self) -> dict[str, float]:
        result = dict(self._usage)
        if self._current_app and self._current_start:
            elapsed = (utc_now() - self._current_start).total_seconds()
            result[self._current_app] = result.get(self._current_app, 0) + elapsed
        return result

    def get_switch_count(self, minutes: int = 30) -> int:
        cutoff = utc_now().timestamp() - (minutes * 60)
        return sum(1 for s in self._switches
                   if datetime.fromisoformat(s["timestamp"]).timestamp() > cutoff)

    def is_deep_work(self, threshold_minutes: int = 45) -> bool:
        if not self._current_app or not self._current_start:
            return False
        elapsed_min = (utc_now() - self._current_start).total_seconds() / 60
        return elapsed_min >= threshold_minutes

    def get_top_apps(self, n: int = 5) -> list[tuple[str, float]]:
        usage = self.get_usage()
        return sorted(usage.items(), key=lambda x: x[1], reverse=True)[:n]

    def reset(self) -> None:
        self._usage.clear()
        self._switches.clear()
        self._current_app = None
        self._current_start = None
