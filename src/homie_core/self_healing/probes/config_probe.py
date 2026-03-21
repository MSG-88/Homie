"""Health probe for configuration validity."""

import os
from pathlib import Path

import yaml

from .base import BaseProbe, HealthStatus, ProbeResult


class ConfigProbe(BaseProbe):
    """Checks configuration file parsability and value validity."""

    name = "config"
    interval = 30.0

    def __init__(self, config, config_path: Path | str) -> None:
        self._config = config
        self._config_path = Path(config_path)

    def check(self) -> ProbeResult:
        errors = []
        metadata = {}

        # Check config file exists and is parseable
        if not self._config_path.exists():
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=f"Config file not found: {self._config_path}",
            )

        try:
            with open(self._config_path) as f:
                yaml.safe_load(f)
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=f"Config parse error: {exc}",
            )

        # Check critical config values
        backend = getattr(self._config.llm, "backend", None)
        model_path = getattr(self._config.llm, "model_path", None)
        metadata["backend"] = backend

        if backend == "gguf" and model_path:
            if not os.path.exists(model_path):
                errors.append(f"Model file not found: {model_path}")

        status = HealthStatus.HEALTHY
        if errors:
            status = HealthStatus.DEGRADED

        return ProbeResult(
            status=status,
            latency_ms=0,
            error_count=len(errors),
            last_error=errors[-1] if errors else None,
            metadata=metadata,
        )
