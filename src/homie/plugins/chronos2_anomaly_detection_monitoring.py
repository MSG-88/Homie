"""Homie plugin for zero-shot time series forecasting and anomaly detection
using Amazon Chronos-2.

Chronos-2 is a pretrained probabilistic time series forecasting model built on
a T5 backbone.  It generates forecasts from raw numeric sequences without any
fine-tuning (zero-shot).  This plugin collects local system metrics (CPU, RAM,
disk, GPU), feeds historical windows to Chronos-2, and flags anomalies when
observed values fall outside the predicted confidence interval.

All inference runs locally via the ``chronos-forecasting`` library and
Hugging Face ``transformers`` / ``torch``.  No network calls are made after
the initial (opt-in) model download.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import numpy as np
    from chronos import ChronosPipeline  # type: ignore[import-untyped]
except ImportError as _imp_err:  # pragma: no cover
    raise ImportError(
        "chronos2_anomaly_detection_monitoring requires "
        "'torch', 'numpy', and 'chronos-forecasting'.  "
        "Install with: pip install torch numpy chronos-forecasting"
    ) from _imp_err

from homie.config import HomieConfig, cfg_get
from homie.controller.notifier import notify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "amazon/chronos-t5-tiny"
DEFAULT_HISTORY_LEN = 64
DEFAULT_PREDICTION_LEN = 8
DEFAULT_CONFIDENCE_LEVEL = 0.90
DEFAULT_POLL_INTERVAL_SEC = 60
DEFAULT_DEVICE = "cpu"
DEFAULT_TORCH_DTYPE = "float32"


# ---------------------------------------------------------------------------
# Local metric collectors (no network calls)
# ---------------------------------------------------------------------------

def _collect_cpu_percent() -> float:
    """Return current CPU utilisation percentage (0-100)."""
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.cpu_percent(interval=0.5)
    except ImportError:
        return _cpu_from_proc()


def _cpu_from_proc() -> float:
    """Fallback CPU usage from /proc/stat (Linux only)."""
    try:
        stat = Path("/proc/stat").read_text(encoding="utf-8")
        parts = stat.splitlines()[0].split()[1:]
        vals = [int(v) for v in parts]
        idle = vals[3]
        total = sum(vals)
        return round(100.0 * (1.0 - idle / max(total, 1)), 2)
    except Exception:  # noqa: BLE001
        return 0.0


def _collect_ram_percent() -> float:
    """Return current RAM utilisation percentage (0-100)."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0.0


def _collect_disk_percent(mount: str = "/") -> float:
    """Return disk usage percentage for *mount*."""
    try:
        import psutil
        return psutil.disk_usage(mount).percent
    except ImportError:
        return 0.0


BUILTIN_COLLECTORS: Dict[str, Callable[[], float]] = {
    "cpu_percent": _collect_cpu_percent,
    "ram_percent": _collect_ram_percent,
    "disk_percent": _collect_disk_percent,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnomalyEvent:
    """Represents a single detected anomaly."""
    metric: str
    timestamp: float
    observed: float
    predicted_low: float
    predicted_high: float
    predicted_median: float
    severity: str  # "warning" | "critical"

    def summary(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.metric}: observed={self.observed:.2f} "
            f"outside [{self.predicted_low:.2f}, {self.predicted_high:.2f}] "
            f"(median={self.predicted_median:.2f})"
        )


@dataclass
class MetricBuffer:
    """Ring-buffer for a single metric's history."""
    name: str
    history: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_LEN))

    def append(self, value: float) -> None:
        self.history.append(value)

    def as_tensor(self) -> "torch.Tensor":
        return torch.tensor(list(self.history), dtype=torch.float32)

    @property
    def ready(self) -> bool:
        return len(self.history) >= 16  # minimum context for reasonable forecasts


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class Chronos2AnomalyPlugin:
    """Zero-shot anomaly detection for local system metrics using Chronos-2.

    Configuration is read from the Homie config under the
    ``plugins.chronos2_anomaly`` key::

        plugins:
          chronos2_anomaly:
            enabled: true
            model_id: amazon/chronos-t5-tiny  # or chronos-t5-small / -base
            device: cpu        # or cuda / mps
            torch_dtype: float32
            history_length: 64
            prediction_length: 8
            confidence_level: 0.90
            poll_interval_sec: 60
            metrics:
              - cpu_percent
              - ram_percent
              - disk_percent
            critical_threshold: 0.98  # confidence band for critical severity
            notify: true
    """

    def __init__(self, cfg: HomieConfig) -> None:
        self.cfg = cfg
        self._plugin_cfg: Dict[str, Any] = (
            cfg_get(cfg, "plugins", "chronos2_anomaly", default={}) or {}
        )

        self.model_id: str = self._plugin_cfg.get("model_id", DEFAULT_MODEL_ID)
        self.device: str = self._plugin_cfg.get("device", DEFAULT_DEVICE)
        self.torch_dtype_str: str = self._plugin_cfg.get("torch_dtype", DEFAULT_TORCH_DTYPE)
        self.history_length: int = int(self._plugin_cfg.get("history_length", DEFAULT_HISTORY_LEN))
        self.prediction_length: int = int(self._plugin_cfg.get("prediction_length", DEFAULT_PREDICTION_LEN))
        self.confidence_level: float = float(self._plugin_cfg.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL))
        self.critical_threshold: float = float(self._plugin_cfg.get("critical_threshold", 0.98))
        self.poll_interval: int = int(self._plugin_cfg.get("poll_interval_sec", DEFAULT_POLL_INTERVAL_SEC))
        self.should_notify: bool = bool(self._plugin_cfg.get("notify", True))

        metric_names: List[str] = self._plugin_cfg.get("metrics", list(BUILTIN_COLLECTORS.keys()))
        self._buffers: Dict[str, MetricBuffer] = {
            m: MetricBuffer(name=m, history=deque(maxlen=self.history_length))
            for m in metric_names
        }
        self._collectors: Dict[str, Callable[[], float]] = {
            m: BUILTIN_COLLECTORS[m] for m in metric_names if m in BUILTIN_COLLECTORS
        }
        self._custom_collectors: Dict[str, Callable[[], float]] = {}

        self._pipeline: Optional[ChronosPipeline] = None
        self._active: bool = False
        self._anomaly_log: List[AnomalyEvent] = []

    # -- lifecycle -----------------------------------------------------------

    def activate(self) -> None:
        """Load the Chronos-2 model and begin monitoring."""
        if self._active:
            logger.warning("Chronos2AnomalyPlugin is already active")
            return

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float32)

        logger.info(
            "Loading Chronos-2 model %s on %s (%s)",
            self.model_id, self.device, self.torch_dtype_str,
        )
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch_dtype,
        )
        self._active = True
        logger.info("Chronos2AnomalyPlugin activated")

    def deactivate(self) -> None:
        """Release the model and stop monitoring."""
        self._active = False
        self._pipeline = None
        logger.info("Chronos2AnomalyPlugin deactivated")

    # -- public API ----------------------------------------------------------

    def register_collector(self, name: str, fn: Callable[[], float]) -> None:
        """Register a custom metric collector.

        Parameters
        ----------
        name:
            Metric name (must be unique).
        fn:
            Callable returning a single float value.
        """
        self._custom_collectors[name] = fn
        if name not in self._buffers:
            self._buffers[name] = MetricBuffer(
                name=name,
                history=deque(maxlen=self.history_length),
            )

    def collect(self) -> Dict[str, float]:
        """Sample all registered metrics and append to history buffers.

        Returns the collected values.
        """
        values: Dict[str, float] = {}
        all_collectors = {**self._collectors, **self._custom_collectors}
        for name, fn in all_collectors.items():
            try:
                val = fn()
                self._buffers[name].append(val)
                values[name] = val
            except Exception:  # noqa: BLE001
                logger.debug("Collector %s failed", name, exc_info=True)
        return values

    def forecast(self, metric: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate a probabilistic forecast for *metric*.

        Returns ``(low, median, high)`` arrays of length ``prediction_length``,
        or ``None`` if insufficient history.
        """
        if self._pipeline is None:
            raise RuntimeError("Plugin not activated â€” call activate() first")

        buf = self._buffers.get(metric)
        if buf is None or not buf.ready:
            return None

        context = buf.as_tensor().unsqueeze(0)  # (1, T)

        quantiles, mean = self._pipeline.predict_quantiles(
            context,
            prediction_length=self.prediction_length,
            quantile_levels=[
                round((1 - self.confidence_level) / 2, 4),
                0.5,
                round(1 - (1 - self.confidence_level) / 2, 4),
            ],
        )
        # quantiles shape: (1, prediction_length, 3)
        q = quantiles[0].numpy()  # (prediction_length, 3)
        low = q[:, 0]
        median = q[:, 1]
        high = q[:, 2]
        return low, median, high

    def detect_anomalies(self) -> List[AnomalyEvent]:
        """Collect fresh metrics, forecast, and return any anomalies."""
        current = self.collect()
        anomalies: List[AnomalyEvent] = []

        for metric, observed in current.items():
            result = self.forecast(metric)
            if result is None:
                continue
            low, median, high = result
            pred_low = float(low[0])
            pred_high = float(high[0])
            pred_median = float(median[0])

            if observed < pred_low or observed > pred_high:
                # Determine severity using a wider critical band
                critical_result = self._forecast_at_confidence(metric, self.critical_threshold)
                severity = "warning"
                if critical_result is not None:
                    c_low, _, c_high = critical_result
                    if observed < float(c_low[0]) or observed > float(c_high[0]):
                        severity = "critical"

                event = AnomalyEvent(
                    metric=metric,
                    timestamp=time.time(),
                    observed=observed,
                    predicted_low=pred_low,
                    predicted_high=pred_high,
                    predicted_median=pred_median,
                    severity=severity,
                )
                anomalies.append(event)
                self._anomaly_log.append(event)
                logger.warning("Anomaly detected: %s", event.summary())

                if self.should_notify:
                    notify(
                        title=f"Homie Anomaly: {metric}",
                        message=event.summary(),
                    )

        return anomalies

    def run_monitor_loop(self, max_iterations: Optional[int] = None) -> None:
        """Blocking loop that periodically collects and checks for anomalies.

        Parameters
        ----------
        max_iterations:
            Stop after this many iterations (useful for testing).
            ``None`` means run indefinitely.
        """
        if not self._active:
            self.activate()

        logger.info(
            "Starting anomaly monitor loop (interval=%ds, metrics=%s)",
            self.poll_interval,
            list(self._buffers.keys()),
        )
        count = 0
        while max_iterations is None or count < max_iterations:
            try:
                anomalies = self.detect_anomalies()
                if anomalies:
                    for a in anomalies:
                        logger.info("  >> %s", a.summary())
            except Exception:  # noqa: BLE001
                logger.exception("Error in monitor iteration")
            count += 1
            if max_iterations is None or count < max_iterations:
                time.sleep(self.poll_interval)

    @property
    def anomaly_history(self) -> List[AnomalyEvent]:
        """Return all anomalies detected since activation."""
        return list(self._anomaly_log)

    # -- internals -----------------------------------------------------------

    def _forecast_at_confidence(
        self, metric: str, confidence: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Forecast with a specific confidence level (used for severity)."""
        if self._pipeline is None:
            return None
        buf = self._buffers.get(metric)
        if buf is None or not buf.ready:
            return None

        context = buf.as_tensor().unsqueeze(0)
        quantiles, _ = self._pipeline.predict_quantiles(
            context,
            prediction_length=self.prediction_length,
            quantile_levels=[
                round((1 - confidence) / 2, 4),
                0.5,
                round(1 - (1 - confidence) / 2, 4),
            ],
        )
        q = quantiles[0].numpy()
        return q[:, 0], q[:, 1], q[:, 2]


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

def activate(cfg: HomieConfig) -> Chronos2AnomalyPlugin:
    """Create and activate the plugin from a Homie config."""
    plugin = Chronos2AnomalyPlugin(cfg)
    plugin.activate()
    return plugin


def deactivate(plugin: Chronos2AnomalyPlugin) -> None:
    """Deactivate and release resources."""
    plugin.deactivate()


__all__ = [
    "Chronos2AnomalyPlugin",
    "AnomalyEvent",
    "MetricBuffer",
    "activate",
    "deactivate",
    "BUILTIN_COLLECTORS",
]
