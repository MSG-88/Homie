"""Behavioral profiler — learns work patterns from observations."""

import threading
from collections import defaultdict
from typing import Optional


class BehavioralProfiler:
    """Learns user's behavioral patterns through observation aggregation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {(hour, category): {value: count}}
        self._patterns: dict[tuple[int, str], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def record_observation(self, hour: int, category: str, value: str) -> None:
        """Record a behavioral observation."""
        with self._lock:
            self._patterns[(hour, category)][value] += 1

    def get_pattern(self, hour: int, category: str) -> dict[str, int]:
        """Get the frequency pattern for a specific hour and category."""
        with self._lock:
            return dict(self._patterns.get((hour, category), {}))

    def predict(self, hour: int, category: str) -> Optional[str]:
        """Predict the most likely value for a given hour and category."""
        with self._lock:
            pattern = self._patterns.get((hour, category))
            if not pattern:
                return None
            return max(pattern, key=pattern.get)

    def get_work_hours(self, min_observations: int = 3) -> list[int]:
        """Determine which hours are typically work hours."""
        work_hours = []
        with self._lock:
            for (hour, category), values in self._patterns.items():
                if category != "activity":
                    continue
                total = sum(values.values())
                work_count = sum(v for k, v in values.items() if k in ("coding", "working", "meeting"))
                if total >= min_observations and work_count > total * 0.5:
                    work_hours.append(hour)
        return sorted(set(work_hours))

    def get_daily_summary(self) -> dict[int, dict[str, str]]:
        """Get a summary of predicted patterns per hour."""
        summary = {}
        with self._lock:
            hours = set(h for (h, _) in self._patterns.keys())
        for hour in sorted(hours):
            summary[hour] = {}
            with self._lock:
                categories = set(c for (h, c) in self._patterns.keys() if h == hour)
            for cat in categories:
                pred = self.predict(hour, cat)
                if pred:
                    summary[hour][cat] = pred
        return summary
