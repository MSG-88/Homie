"""Preference engine with EMA scoring and CUSUM change-point detection."""

from __future__ import annotations

import json
from typing import Any


class _CUSUMTracker:
    """Tracks cumulative sum for one preference signal to detect shifts."""

    WARMUP = 5

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        self.observations: list[float] = []
        self.mean: float = 0.0
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0
        self.slack: float = 0.0
        self._warmed_up: bool = False

    def update(self, value: float) -> bool:
        """Record a new value. Returns True if a shift is detected."""
        self.observations.append(value)
        n = len(self.observations)

        if n <= self.WARMUP:
            # During warmup, just accumulate and compute baseline stats.
            self.mean = sum(self.observations) / n
            if n == self.WARMUP:
                variance = sum(
                    (x - self.mean) ** 2 for x in self.observations
                ) / n
                std = max(variance ** 0.5, 0.01)
                self.slack = std * 0.5
                self._warmed_up = True
            return False

        # Post-warmup: run CUSUM
        deviation = value - self.mean
        self.cusum_pos = max(0.0, self.cusum_pos + deviation - self.slack)
        self.cusum_neg = max(0.0, self.cusum_neg - deviation - self.slack)

        if self.cusum_pos > self.threshold or self.cusum_neg > self.threshold:
            # Shift detected -- reset tracker with new baseline
            self.cusum_pos = 0.0
            self.cusum_neg = 0.0
            # Recalculate mean from recent observations
            recent = self.observations[-self.WARMUP:]
            self.mean = sum(recent) / len(recent)
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "observations": self.observations,
            "mean": self.mean,
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "slack": self.slack,
            "_warmed_up": self._warmed_up,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _CUSUMTracker:
        tracker = cls(threshold=data["threshold"])
        tracker.observations = data["observations"]
        tracker.mean = data["mean"]
        tracker.cusum_pos = data["cusum_pos"]
        tracker.cusum_neg = data["cusum_neg"]
        tracker.slack = data["slack"]
        tracker._warmed_up = data["_warmed_up"]
        return tracker


class PreferenceEngine:
    """Learns user preferences using EMA scores with CUSUM change-point detection."""

    def __init__(
        self, ema_alpha: float = 0.1, cusum_threshold: float = 3.0
    ) -> None:
        self.ema_alpha = ema_alpha
        self.cusum_threshold = cusum_threshold
        # domain -> key -> EMA score
        self._preferences: dict[str, dict[str, float]] = {}
        # domain -> key -> CUSUM tracker
        self._trackers: dict[str, dict[str, _CUSUMTracker]] = {}
        # List of detected shift events
        self._shifts: list[dict[str, Any]] = []
        # domain -> key -> observation count (for EMA init)
        self._counts: dict[str, dict[str, int]] = {}

    def record(self, domain: str, key: str, score: float) -> None:
        """Record a preference observation and update EMA + CUSUM."""
        # Initialise structures if needed
        if domain not in self._preferences:
            self._preferences[domain] = {}
            self._trackers[domain] = {}
            self._counts[domain] = {}

        if key not in self._preferences[domain]:
            self._preferences[domain][key] = 0.0
            self._trackers[domain][key] = _CUSUMTracker(
                threshold=self.cusum_threshold
            )
            self._counts[domain][key] = 0

        # Update EMA
        self._counts[domain][key] += 1
        count = self._counts[domain][key]
        if count == 1:
            self._preferences[domain][key] = score
        else:
            prev = self._preferences[domain][key]
            self._preferences[domain][key] = (
                self.ema_alpha * score + (1 - self.ema_alpha) * prev
            )

        # Update CUSUM tracker
        shift_detected = self._trackers[domain][key].update(score)
        if shift_detected:
            self._shifts.append(
                {
                    "domain": domain,
                    "key": key,
                    "observation": count,
                    "current_ema": self._preferences[domain][key],
                }
            )

    def _strength(self, domain: str, key: str) -> float:
        """Compute preference strength: EMA weighted by confidence."""
        ema = self._preferences[domain][key]
        count = self._counts[domain][key]
        # Confidence ramps from 0 toward 1 as observations grow
        confidence = count / (count + 1.0)
        return ema * confidence

    def get_preferences(self, domain: str) -> dict[str, float]:
        """Return preference scores for a domain."""
        prefs = self._preferences.get(domain, {})
        return {
            key: self._strength(domain, key) for key in prefs
        }

    def get_all_preferences(self) -> dict[str, dict[str, float]]:
        """Return all preference scores across all domains."""
        return {
            domain: self.get_preferences(domain)
            for domain in self._preferences
        }

    def get_dominant(self, domain: str) -> str | None:
        """Return the key with the highest preference in a domain."""
        prefs = self.get_preferences(domain)
        if not prefs:
            return None
        return max(prefs, key=prefs.get)  # type: ignore[arg-type]

    def get_detected_shifts(self) -> list[dict[str, Any]]:
        """Return all detected preference shifts."""
        return list(self._shifts)

    def serialize(self) -> str:
        """Serialize engine state to a JSON string."""
        data = {
            "ema_alpha": self.ema_alpha,
            "cusum_threshold": self.cusum_threshold,
            "preferences": self._preferences,
            "counts": self._counts,
            "trackers": {
                domain: {
                    key: tracker.to_dict()
                    for key, tracker in keys.items()
                }
                for domain, keys in self._trackers.items()
            },
            "shifts": self._shifts,
        }
        return json.dumps(data)

    @classmethod
    def deserialize(cls, data: str) -> PreferenceEngine:
        """Restore engine state from a JSON string."""
        obj = json.loads(data)
        engine = cls(
            ema_alpha=obj["ema_alpha"],
            cusum_threshold=obj["cusum_threshold"],
        )
        engine._preferences = obj["preferences"]
        engine._counts = obj["counts"]
        engine._shifts = obj["shifts"]
        engine._trackers = {
            domain: {
                key: _CUSUMTracker.from_dict(tdata)
                for key, tdata in keys.items()
            }
            for domain, keys in obj["trackers"].items()
        }
        return engine
