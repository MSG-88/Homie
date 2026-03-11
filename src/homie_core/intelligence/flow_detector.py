"""Shannon entropy-based flow state detector."""

from __future__ import annotations

import math
from collections import Counter, deque


class FlowDetector:
    """Detects user flow state using Shannon entropy over a sliding window of activities."""

    def __init__(self, window_size: int = 30, flow_threshold: float = 0.7) -> None:
        self._window: deque[str] = deque(maxlen=window_size)
        self._flow_threshold = flow_threshold

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_activity(self, activity: str) -> None:
        """Append an activity label to the sliding window."""
        self._window.append(activity)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_entropy(self) -> float:
        """Shannon entropy H = -sum(p_i * log2(p_i)) over activity proportions."""
        n = len(self._window)
        if n == 0:
            return 0.0
        counts = Counter(self._window)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def get_switch_rate(self) -> float:
        """Fraction of consecutive activity pairs that differ."""
        n = len(self._window)
        if n < 2:
            return 0.0
        switches = sum(
            1 for a, b in zip(list(self._window), list(self._window)[1:]) if a != b
        )
        return switches / (n - 1)

    def get_flow_score(self) -> float:
        """Composite flow score: 0.5 * (1 - norm_entropy) + 0.5 * (1 - switch_rate).

        Entropy is normalised by log2(n_unique) so that it falls in [0, 1].
        Returns 0.5 when the window is empty (neutral).
        """
        n = len(self._window)
        if n == 0:
            return 0.5

        n_unique = len(set(self._window))
        entropy = self.compute_entropy()

        if n_unique <= 1:
            norm_entropy = 0.0
        else:
            norm_entropy = entropy / math.log2(n_unique)

        switch_rate = self.get_switch_rate()
        return 0.5 * (1.0 - norm_entropy) + 0.5 * (1.0 - switch_rate)

    def is_in_flow(self) -> bool:
        """Return True when flow_score >= threshold."""
        return self.get_flow_score() >= self._flow_threshold

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_focus_report(self) -> dict:
        """Return a summary dict of the current flow state."""
        counts = Counter(self._window)
        dominant = counts.most_common(1)[0][0] if counts else None
        return {
            "entropy": self.compute_entropy(),
            "flow_score": self.get_flow_score(),
            "switch_rate": self.get_switch_rate(),
            "dominant_activity": dominant,
            "unique_activities": len(set(self._window)),
            "window_fill": len(self._window),
            "in_flow": self.is_in_flow(),
        }
