"""Health probe for the inference pipeline."""

from .base import BaseProbe, HealthStatus, ProbeResult


class InferenceProbe(BaseProbe):
    """Checks model engine health and inference responsiveness."""

    name = "inference"
    interval = 10.0  # critical — check every 10s

    def __init__(self, model_engine, inference_router) -> None:
        self._engine = model_engine
        self._router = inference_router

    def check(self) -> ProbeResult:
        source = self._router.active_source
        errors = 0
        last_error = None

        if not self._engine.is_loaded:
            return ProbeResult(
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error_count=1,
                last_error="Model not loaded",
                metadata={"source": source},
            )

        # Test a minimal generation
        try:
            self._engine.generate("ping", max_tokens=1, timeout=10)
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=str(exc),
                metadata={"source": source},
            )

        # Check if we're on a fallback source
        status = HealthStatus.HEALTHY
        if source != "Local":
            status = HealthStatus.DEGRADED

        return ProbeResult(
            status=status,
            latency_ms=0,
            error_count=errors,
            last_error=last_error,
            metadata={"source": source},
        )
