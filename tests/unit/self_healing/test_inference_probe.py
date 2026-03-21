# tests/unit/self_healing/test_inference_probe.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.probes.inference_probe import InferenceProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestInferenceProbe:
    def _make_probe(self, engine=None, router=None):
        engine = engine or MagicMock()
        router = router or MagicMock()
        return InferenceProbe(model_engine=engine, inference_router=router)

    def test_healthy_when_model_loaded_and_responds(self):
        engine = MagicMock()
        engine.is_loaded = True
        engine.generate.return_value = "hello"
        router = MagicMock()
        router.active_source = "Local"
        probe = self._make_probe(engine=engine, router=router)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.metadata["source"] == "Local"

    def test_degraded_when_model_not_loaded(self):
        engine = MagicMock()
        engine.is_loaded = False
        router = MagicMock()
        router.active_source = "None"
        probe = self._make_probe(engine=engine, router=router)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_generate_raises(self):
        engine = MagicMock()
        engine.is_loaded = True
        engine.generate.side_effect = RuntimeError("OOM")
        router = MagicMock()
        router.active_source = "Local"
        probe = self._make_probe(engine=engine, router=router)
        result = probe.check()
        assert result.status == HealthStatus.FAILED
        assert "OOM" in result.last_error

    def test_degraded_when_on_fallback_source(self):
        engine = MagicMock()
        engine.is_loaded = True
        engine.generate.return_value = "ok"
        router = MagicMock()
        router.active_source = "Homie Intelligence (Cloud)"
        probe = self._make_probe(engine=engine, router=router)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED
