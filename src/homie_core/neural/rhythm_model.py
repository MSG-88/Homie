"""Fourier-based circadian rhythm modeling.

Uses Discrete Fourier Transform to decompose productivity signals into
frequency components, identifying daily/weekly cycles and predicting
optimal work windows.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional


class CircadianRhythmModel:
    """Models user productivity as a sum of sinusoidal components.

    Records productivity observations bucketed by hour-of-day,
    decomposes the signal via DFT, and reconstructs a continuous
    productivity curve for prediction.
    """

    def __init__(self, num_hours: int = 24):
        self._num_hours = num_hours
        self._hourly_buckets: dict[int, list[float]] = defaultdict(list)
        self._activity_buckets: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._fourier_cache: Optional[list[dict]] = None

    def record_activity(
        self,
        hour: int,
        productivity_score: float,
        activity_type: Optional[str] = None,
    ) -> None:
        """Record a productivity observation at a given hour."""
        hour = hour % self._num_hours
        score = max(0.0, min(1.0, productivity_score))
        self._hourly_buckets[hour].append(score)
        if activity_type:
            self._activity_buckets[activity_type][hour].append(score)
        self._fourier_cache = None  # invalidate

    def get_hourly_averages(self) -> list[float]:
        """Return average productivity for each hour slot."""
        result = []
        for h in range(self._num_hours):
            bucket = self._hourly_buckets[h]
            result.append(sum(bucket) / len(bucket) if bucket else 0.0)
        return result

    def _dft(self, signal: list[float]) -> list[tuple[float, float]]:
        """Compute Discrete Fourier Transform.

        Returns list of (real, imag) pairs for each frequency bin.
        """
        n = len(signal)
        result = []
        for k in range(n):
            re = 0.0
            im = 0.0
            for t in range(n):
                angle = 2 * math.pi * k * t / n
                re += signal[t] * math.cos(angle)
                im -= signal[t] * math.sin(angle)
            result.append((re, im))
        return result

    def fourier_decompose(self, top_k: int = 5) -> list[dict]:
        """Decompose hourly productivity into frequency components.

        Returns top_k components sorted by amplitude, each with:
        - frequency: cycles per day (for 24h signal)
        - amplitude: strength of this component
        - phase: phase offset in radians
        - period: period in hours
        """
        if self._fourier_cache is not None:
            return self._fourier_cache[:top_k]

        signal = self.get_hourly_averages()
        n = len(signal)
        if all(v == 0.0 for v in signal):
            return []

        dft = self._dft(signal)
        components = []
        for k in range(1, n // 2 + 1):  # skip DC component (k=0)
            re, im = dft[k]
            amplitude = 2.0 * math.sqrt(re * re + im * im) / n
            phase = math.atan2(-im, re)
            period = n / k  # in hours
            components.append({
                "frequency": k,
                "amplitude": amplitude,
                "phase": phase,
                "period": period,
            })

        components.sort(key=lambda c: c["amplitude"], reverse=True)
        self._fourier_cache = components
        return components[:top_k]

    def predict_productivity(self, hour: float) -> float:
        """Predict productivity at a fractional hour using Fourier reconstruction."""
        signal = self.get_hourly_averages()
        n = len(signal)
        if all(v == 0.0 for v in signal):
            return 0.5  # no data, return neutral

        # DC component (mean)
        dft = self._dft(signal)
        dc = dft[0][0] / n

        # Reconstruct from top components
        components = self.fourier_decompose(top_k=6)
        value = dc
        for comp in components:
            k = comp["frequency"]
            value += comp["amplitude"] * math.cos(
                2 * math.pi * k * hour / n - comp["phase"]
            )

        return max(0.0, min(1.0, value))

    def get_optimal_windows(self, top_n: int = 3) -> list[dict]:
        """Find the top_n most productive hours."""
        predictions = []
        for h in range(self._num_hours):
            predictions.append({
                "hour": h,
                "predicted_score": self.predict_productivity(h),
            })
        predictions.sort(key=lambda p: p["predicted_score"], reverse=True)
        return predictions[:top_n]

    def get_activity_rhythm(self) -> dict[str, dict[int, float]]:
        """Return per-activity hourly averages."""
        result = {}
        for activity, buckets in self._activity_buckets.items():
            hourly = {}
            for h, scores in buckets.items():
                if scores:
                    hourly[h] = sum(scores) / len(scores)
            result[activity] = hourly
        return result

    def serialize(self) -> dict:
        """Serialize model state."""
        return {
            "num_hours": self._num_hours,
            "hourly_buckets": {
                str(k): v for k, v in self._hourly_buckets.items()
            },
            "activity_buckets": {
                act: {str(h): scores for h, scores in buckets.items()}
                for act, buckets in self._activity_buckets.items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> CircadianRhythmModel:
        """Deserialize from dict."""
        model = cls(num_hours=data.get("num_hours", 24))
        for k, v in data.get("hourly_buckets", {}).items():
            model._hourly_buckets[int(k)] = v
        for act, buckets in data.get("activity_buckets", {}).items():
            for h, scores in buckets.items():
                model._activity_buckets[act][int(h)] = scores
        return model
