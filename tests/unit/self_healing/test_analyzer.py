# tests/unit/self_healing/test_analyzer.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.improvement.analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    def test_profile_module_returns_metrics(self):
        metrics = MagicMock()
        metrics.snapshot.return_value = {
            "inference": {
                "latency_ms": {"latest": 200, "average": 150, "count": 50},
                "error_count": {"latest": 0, "average": 0.1, "count": 50},
            }
        }
        analyzer = PerformanceAnalyzer(metrics=metrics)
        profile = analyzer.profile("inference")
        assert "latency_ms" in profile
        assert profile["latency_ms"]["latest"] == 200

    def test_identify_bottlenecks(self):
        metrics = MagicMock()
        metrics.snapshot.return_value = {
            "inference": {"latency_ms": {"latest": 5000, "average": 200, "count": 50}},
            "storage": {"latency_ms": {"latest": 10, "average": 8, "count": 50}},
        }
        analyzer = PerformanceAnalyzer(metrics=metrics)
        bottlenecks = analyzer.identify_bottlenecks()
        assert len(bottlenecks) >= 1
        assert bottlenecks[0]["module"] == "inference"

    def test_no_bottlenecks_when_healthy(self):
        metrics = MagicMock()
        metrics.snapshot.return_value = {
            "inference": {"latency_ms": {"latest": 100, "average": 100, "count": 50}},
        }
        analyzer = PerformanceAnalyzer(metrics=metrics)
        bottlenecks = analyzer.identify_bottlenecks()
        assert len(bottlenecks) == 0

    def test_trend_detection_increasing(self):
        metrics = MagicMock()
        # Simulate an increasing trend by returning high latest vs average
        metrics.snapshot.return_value = {
            "inference": {"latency_ms": {"latest": 500, "average": 100, "count": 50}},
        }
        analyzer = PerformanceAnalyzer(metrics=metrics, trend_threshold=2.0)
        trends = analyzer.detect_trends()
        assert len(trends) >= 1
