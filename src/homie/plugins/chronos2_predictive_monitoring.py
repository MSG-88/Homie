"""Homie Predictive Monitoring Plugin using Amazon Chronos-2.

Provides zero-shot time-series forecasting for system metrics (CPU, memory,
disk usage, temperatures) collected by Homie nodes. Uses the Chronos-2 model
family via HuggingFace transformers to predict future resource consumption
and emit alerts before thresholds are breached.

All inference runs locally — no network calls required after initial model download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default model suitable for CPU inference on consumer hardware
DEFAULT_MODEL_ID = "amazon/chronos-t5-small"
DEFAULT_PREDICTION_HORIZON = 12  # steps ahead
DEFAULT_CONTEXT_LENGTH = 64  # historical observations fed to the model
DEFAULT_ALERT_THRESHOLD_PCT = 85.0


@dataclass
class ForecastResult:
    """Container for a single metric forecast."""

    metric_name: str
    target: str
    timestamp: str
    horizon_steps: int
    predicted_mean: List[float]
    predicted_low: List[float]
    predicted_high: List[float]
    breach_step: Optional[int] = None
    breach_value: Optional[float] = None
    alert: bool = False
    message: str = ""


@dataclass
class PredictiveMonitoringConfig:
    """Plugin configuration, loadable from homie.config.yaml."""

    enabled: bool = True
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON
    context_length: int = DEFAULT_CONTEXT_LENGTH
    alert_threshold_pct: float = DEFAULT_ALERT_THRESHOLD_PCT
    check_interval_sec: int = 300
    metrics: List[str] = field(default_factory=lambda: ["cpu_pct", "mem_pct", "disk_pct"])
    cache_dir: Optional[str] = None


class Chronos2Forecaster:
    """Wraps Chronos-2 pipeline for local time-series prediction."""

    def __init__(self, config: PredictiveMonitoringConfig) -> None:
        self.config = config
        self._pipeline = None

    def _load_pipeline(self) -> None:
        """Lazy-load the Chronos pipeline on first use."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise RuntimeError(
                "chronos-forecasting and torch are required. "
                "Install with: pip install chronos-forecasting torch"
            ) from exc

        logger.info("Loading Chronos-2 model: %s on %s", self.config.model_id, self.config.device)

        kwargs: Dict[str, Any] = {
            "device_map": self.config.device,
            "torch_dtype": torch.float32,
        }
        if self.config.cache_dir:
            kwargs["cache_dir"] = self.config.cache_dir

        self._pipeline = ChronosPipeline.from_pretrained(
            self.config.model_id,
            **kwargs,
        )
        logger.info("Chronos-2 model loaded successfully.")

    def forecast(
        self,
        context: Sequence[float],
        prediction_length: Optional[int] = None,
        num_samples: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate probabilistic forecast from historical context.

        Args:
            context: Historical metric values (most recent last).
            prediction_length: Steps to forecast. Defaults to config horizon.
            num_samples: Number of sample paths for uncertainty estimation.

        Returns:
            Tuple of (mean, low_quantile, high_quantile) arrays.
        """
        import torch

        self._load_pipeline()

        horizon = prediction_length or self.config.prediction_horizon
        ctx_tensor = torch.tensor(list(context), dtype=torch.float32)

        # Chronos returns shape (num_series, num_samples, prediction_length)
        samples = self._pipeline.predict(
            context=ctx_tensor.unsqueeze(0),
            prediction_length=horizon,
            num_samples=num_samples,
        )

        # Compute quantiles across samples
        samples_np = samples.numpy().squeeze(0)  # (num_samples, horizon)
        mean = np.median(samples_np, axis=0)
        low = np.quantile(samples_np, 0.1, axis=0)
        high = np.quantile(samples_np, 0.9, axis=0)

        return mean, low, high


class PredictiveMonitoringPlugin:
    """Homie plugin that forecasts system metrics and raises predictive alerts.

    Integration points:
    - Reads metric history from Homie storage (SQLite via controller.storage)
    - Emits notifications via controller.notifier when breaches are predicted
    - Registers a scheduled job via controller.scheduler for periodic checks
    """

    def __init__(self, config: Optional[PredictiveMonitoringConfig] = None) -> None:
        self.config = config or PredictiveMonitoringConfig()
        self._forecaster: Optional[Chronos2Forecaster] = None
        self._active = False

    @property
    def forecaster(self) -> Chronos2Forecaster:
        if self._forecaster is None:
            self._forecaster = Chronos2Forecaster(self.config)
        return self._forecaster

    def activate(self, scheduler=None, storage=None, notifier=None) -> None:
        """Activate the plugin, optionally registering with Homie scheduler.

        Args:
            scheduler: An AutomationScheduler instance for periodic forecasting.
            storage: A Storage instance for reading metric history.
            notifier: A notify callable for sending alerts.
        """
        if not self.config.enabled:
            logger.info("PredictiveMonitoring plugin is disabled via config.")
            return

        self._storage = storage
        self._notifier = notifier
        self._active = True

        if scheduler is not None:
            interval_min = max(1, self.config.check_interval_sec // 60)
            cron_expr = f"*/{interval_min} * * * *"
            scheduler.add_cron_job(
                name="predictive_monitoring_check",
                cron=cron_expr,
                func=self._run_check,
                args=[],
                kwargs={},
            )
            logger.info(
                "PredictiveMonitoring scheduled every %d minutes.", interval_min
            )

        logger.info("PredictiveMonitoring plugin activated.")

    def deactivate(self, scheduler=None) -> None:
        """Deactivate the plugin and remove scheduled jobs."""
        self._active = False
        if scheduler is not None:
            scheduler.remove("predictive_monitoring_check")
        self._forecaster = None
        logger.info("PredictiveMonitoring plugin deactivated.")

    def predict_metric(
        self,
        metric_name: str,
        target: str,
        history: Sequence[float],
    ) -> ForecastResult:
        """Run a forecast for a single metric series.

        Args:
            metric_name: Name of the metric (e.g. 'cpu_pct').
            target: Machine target identifier.
            history: Recent metric observations, oldest first.

        Returns:
            ForecastResult with predictions and optional alert.
        """
        # Trim to context length
        ctx = list(history[-self.config.context_length:])

        if len(ctx) < 4:
            return ForecastResult(
                metric_name=metric_name,
                target=target,
                timestamp=datetime.utcnow().isoformat(),
                horizon_steps=self.config.prediction_horizon,
                predicted_mean=[],
                predicted_low=[],
                predicted_high=[],
                message="Insufficient history for forecasting.",
            )

        mean, low, high = self.forecaster.forecast(ctx)

        # Check for threshold breach
        breach_step: Optional[int] = None
        breach_value: Optional[float] = None
        threshold = self.config.alert_threshold_pct

        for i, val in enumerate(mean):
            if val >= threshold:
                breach_step = i
                breach_value = float(val)
                break

        alert = breach_step is not None
        message = ""
        if alert:
            message = (
                f"PREDICTED: {metric_name} on {target} will reach "
                f"{breach_value:.1f}% in ~{breach_step + 1} step(s). "
                f"Threshold: {threshold}%."
            )

        return ForecastResult(
            metric_name=metric_name,
            target=target,
            timestamp=datetime.utcnow().isoformat(),
            horizon_steps=self.config.prediction_horizon,
            predicted_mean=[float(v) for v in mean],
            predicted_low=[float(v) for v in low],
            predicted_high=[float(v) for v in high],
            breach_step=breach_step,
            breach_value=breach_value,
            alert=alert,
            message=message,
        )

    def _run_check(self) -> List[ForecastResult]:
        """Periodic check: forecast all configured metrics for all machines."""
        results: List[ForecastResult] = []

        if not self._active:
            return results

        if self._storage is None:
            logger.warning("No storage attached; skipping predictive check.")
            return results

        machines = self._storage.list_machines()
        for machine in machines:
            target = machine.get("display_name") or machine.get("ip", "unknown")
            for metric_name in self.config.metrics:
                history = self._fetch_metric_history(machine, metric_name)
                if not history:
                    continue

                result = self.predict_metric(metric_name, target, history)
                results.append(result)

                if result.alert and self._notifier:
                    self._notifier(
                        title="Predictive Alert",
                        message=result.message,
                    )

        return results

    def _fetch_metric_history(
        self, machine: Dict[str, Any], metric_name: str
    ) -> List[float]:
        """Extract metric time-series from storage.

        Override or extend this method for custom metric sources.
        """
        # Default: query recent runs stdout for numeric patterns.
        # In production, this would read from a dedicated metrics table or
        # time-series store. Placeholder returns empty to be safe.
        return []


def register(cfg_raw: Optional[Dict[str, Any]] = None) -> PredictiveMonitoringPlugin:
    """Factory function to create and return a configured plugin instance.

    Args:
        cfg_raw: Optional dict from homie.config.yaml under
                 'plugins.predictive_monitoring'.

    Returns:
        Configured but not yet activated plugin instance.
    """
    config = PredictiveMonitoringConfig()
    if cfg_raw:
        config.enabled = cfg_raw.get("enabled", config.enabled)
        config.model_id = cfg_raw.get("model_id", config.model_id)
        config.device = cfg_raw.get("device", config.device)
        config.prediction_horizon = cfg_raw.get("prediction_horizon", config.prediction_horizon)
        config.context_length = cfg_raw.get("context_length", config.context_length)
        config.alert_threshold_pct = cfg_raw.get("alert_threshold_pct", config.alert_threshold_pct)
        config.check_interval_sec = cfg_raw.get("check_interval_sec", config.check_interval_sec)
        config.metrics = cfg_raw.get("metrics", config.metrics)
        config.cache_dir = cfg_raw.get("cache_dir", config.cache_dir)

    return PredictiveMonitoringPlugin(config=config)


__all__ = [
    "PredictiveMonitoringPlugin",
    "PredictiveMonitoringConfig",
    "Chronos2Forecaster",
    "ForecastResult",
    "register",
]
