"""Tests for PatternDetector — periodicity, anomalies, and trends."""

import math
import pytest

from homie_core.ml.pattern_detector import PatternDetector


@pytest.fixture
def detector():
    return PatternDetector()


# ---------------------------------------------------------------------------
# Periodicity
# ---------------------------------------------------------------------------

class TestDetectPeriodicity:
    def test_regular_periodic_signal(self, detector):
        # Events every 10 units
        timestamps = [10.0 * i for i in range(20)]
        values = ["A", "B"] * 10  # alternating
        result = detector.detect_periodicity(timestamps, values)
        assert result["is_periodic"] is True
        assert result["period"] is not None
        assert result["confidence"] > 0.5

    def test_non_periodic_signal(self, detector):
        timestamps = [1.0, 3.0, 7.0, 15.0, 31.0, 63.0]
        values = ["A"] * 6
        result = detector.detect_periodicity(timestamps, values)
        # exponentially growing gaps -> low confidence
        assert result["confidence"] < 0.8

    def test_too_few_points(self, detector):
        result = detector.detect_periodicity([1.0, 2.0], ["A", "B"])
        assert result["is_periodic"] is False
        assert result["period"] is None

    def test_perfectly_periodic(self, detector):
        timestamps = [float(i) for i in range(0, 100, 5)]
        values = ["X"] * len(timestamps)
        result = detector.detect_periodicity(timestamps, values)
        assert result["is_periodic"] is True
        assert abs(result["period"] - 5.0) < 0.01
        assert result["confidence"] > 0.95

    def test_mixed_values(self, detector):
        # Two different periodic patterns
        timestamps = list(range(30))
        values = ["A" if i % 3 == 0 else "B" for i in range(30)]
        result = detector.detect_periodicity(timestamps, values)
        assert result["is_periodic"] is True


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class TestDetectAnomalies:
    def test_obvious_anomaly(self, detector):
        values = [10.0] * 20 + [100.0]  # spike at index 20
        anomalies = detector.detect_anomalies(values)
        assert 20 in anomalies

    def test_no_anomalies_in_uniform_data(self, detector):
        values = [5.0] * 30
        assert detector.detect_anomalies(values) == []

    def test_custom_threshold(self, detector):
        values = [0.0] * 10 + [3.0]
        # With a very high threshold no anomaly
        assert detector.detect_anomalies(values, threshold=10.0) == []
        # With a low threshold the spike is flagged
        anomalies = detector.detect_anomalies(values, threshold=1.0)
        assert 10 in anomalies

    def test_single_value(self, detector):
        assert detector.detect_anomalies([42.0]) == []

    def test_negative_anomaly(self, detector):
        values = [50.0] * 20 + [-50.0]
        anomalies = detector.detect_anomalies(values, threshold=2.0)
        assert 20 in anomalies


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

class TestDetectTrends:
    def test_increasing(self, detector):
        values = [float(i) for i in range(20)]
        assert detector.detect_trends(values) == "increasing"

    def test_decreasing(self, detector):
        values = [float(20 - i) for i in range(20)]
        assert detector.detect_trends(values) == "decreasing"

    def test_stable(self, detector):
        values = [5.0] * 20
        assert detector.detect_trends(values) == "stable"

    def test_noisy_but_increasing(self, detector):
        # Strong upward trend with a little noise
        import random
        random.seed(42)
        values = [float(i * 10) + random.uniform(-1, 1) for i in range(30)]
        assert detector.detect_trends(values) == "increasing"

    def test_single_value_stable(self, detector):
        assert detector.detect_trends([7.0]) == "stable"


# ---------------------------------------------------------------------------
# Value frequencies (bonus helper)
# ---------------------------------------------------------------------------

class TestValueFrequencies:
    def test_basic(self, detector):
        freq = detector.value_frequencies(["a", "b", "a", "c", "a"])
        assert freq == {"a": 3, "b": 1, "c": 1}

    def test_empty(self, detector):
        assert detector.value_frequencies([]) == {}
