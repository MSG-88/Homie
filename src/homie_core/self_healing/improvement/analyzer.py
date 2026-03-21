"""Performance analyzer — profiles execution and identifies bottlenecks."""

from typing import Any

from ..metrics import MetricsCollector


class PerformanceAnalyzer:
    """Analyzes module performance metrics to find optimization opportunities."""

    def __init__(
        self,
        metrics: MetricsCollector,
        bottleneck_ratio: float = 3.0,
        trend_threshold: float = 2.0,
    ) -> None:
        self._metrics = metrics
        self._bottleneck_ratio = bottleneck_ratio
        self._trend_threshold = trend_threshold

    def profile(self, module: str) -> dict[str, Any]:
        """Get performance profile for a specific module."""
        snapshot = self._metrics.snapshot()
        return snapshot.get(module, {})

    def identify_bottlenecks(self) -> list[dict[str, Any]]:
        """Find modules where latest metrics significantly exceed baseline."""
        bottlenecks = []
        snapshot = self._metrics.snapshot()

        for module, metrics in snapshot.items():
            for metric_name, values in metrics.items():
                latest = values.get("latest", 0)
                average = values.get("average", 0)
                count = values.get("count", 0)

                if count < 10 or average <= 0:
                    continue

                ratio = latest / average
                if ratio >= self._bottleneck_ratio:
                    bottlenecks.append({
                        "module": module,
                        "metric": metric_name,
                        "latest": latest,
                        "average": average,
                        "ratio": ratio,
                    })

        bottlenecks.sort(key=lambda b: b["ratio"], reverse=True)
        return bottlenecks

    def detect_trends(self) -> list[dict[str, Any]]:
        """Detect metrics trending in a concerning direction."""
        trends = []
        snapshot = self._metrics.snapshot()

        for module, metrics in snapshot.items():
            for metric_name, values in metrics.items():
                latest = values.get("latest", 0)
                average = values.get("average", 0)
                count = values.get("count", 0)

                if count < 10 or average <= 0:
                    continue

                if latest > average * self._trend_threshold:
                    trends.append({
                        "module": module,
                        "metric": metric_name,
                        "direction": "increasing",
                        "latest": latest,
                        "average": average,
                    })

        return trends
