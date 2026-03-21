# tests/unit/self_healing/test_config_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.config_probe import ConfigProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestConfigProbe:
    def test_healthy_with_valid_config(self, tmp_path):
        config_path = tmp_path / "homie.config.yaml"
        config_path.write_text("llm:\n  backend: gguf\n")
        config = MagicMock()
        config.llm.backend = "gguf"
        config.llm.model_path = str(tmp_path / "model.gguf")
        # Create a fake model file
        (tmp_path / "model.gguf").touch()
        probe = ConfigProbe(config=config, config_path=config_path)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_model_path_missing(self, tmp_path):
        config_path = tmp_path / "homie.config.yaml"
        config_path.write_text("llm:\n  backend: gguf\n")
        config = MagicMock()
        config.llm.backend = "gguf"
        config.llm.model_path = "/nonexistent/model.gguf"
        probe = ConfigProbe(config=config, config_path=config_path)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_config_file_missing(self, tmp_path):
        config = MagicMock()
        config.llm.backend = "gguf"
        probe = ConfigProbe(config=config, config_path=tmp_path / "missing.yaml")
        result = probe.check()
        assert result.status == HealthStatus.FAILED
