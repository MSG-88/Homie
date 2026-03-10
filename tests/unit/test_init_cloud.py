import yaml
from pathlib import Path
from unittest.mock import patch

from homie_core.config import HomieConfig


def test_cloud_setup_saves_config(tmp_path, monkeypatch):
    """Simulate user choosing cloud setup."""
    monkeypatch.chdir(tmp_path)

    # Simulate user inputs: "2" (cloud), "1" (OpenAI), "sk-test", "1" (first model), "TestUser"
    inputs = iter(["2", "1", "sk-test-key", "1", "TestUser"])

    from homie_app.init import run_init

    with patch("builtins.input", lambda prompt="": next(inputs)):
        with patch("homie_app.init.detect_hardware") as mock_hw:
            mock_hw.return_value.os_name = "Windows"
            mock_hw.return_value.os_version = "11"
            mock_hw.return_value.cpu_cores = 8
            mock_hw.return_value.ram_gb = 32
            mock_hw.return_value.gpus = []
            mock_hw.return_value.has_microphone = False
            mock_hw.return_value.best_gpu_vram_gb = 0

            with patch("homie_app.init._discover_cloud_models", return_value=["gpt-4o", "gpt-4o-mini"]):
                cfg = run_init(auto=False)

    assert cfg.llm.backend == "cloud"
    assert cfg.llm.api_key == "sk-test-key"
    assert cfg.llm.model_path == "gpt-4o"
    assert cfg.user_name == "TestUser"

    # Verify YAML was written
    config_file = tmp_path / "homie.config.yaml"
    assert config_file.exists()
    saved = yaml.safe_load(config_file.read_text())
    assert saved["llm"]["backend"] == "cloud"


def test_local_setup_unchanged(tmp_path, monkeypatch):
    """Simulate user choosing local setup — existing flow should still work."""
    monkeypatch.chdir(tmp_path)

    inputs = iter(["1", "TestUser"])

    from homie_app.init import run_init

    with patch("builtins.input", lambda prompt="": next(inputs)):
        with patch("homie_app.init.detect_hardware") as mock_hw:
            mock_hw.return_value.os_name = "Windows"
            mock_hw.return_value.os_version = "11"
            mock_hw.return_value.cpu_cores = 8
            mock_hw.return_value.ram_gb = 32
            mock_hw.return_value.gpus = []
            mock_hw.return_value.has_microphone = False
            mock_hw.return_value.best_gpu_vram_gb = 0

            with patch("homie_app.init.discover_local_model", return_value=None):
                cfg = run_init(auto=False)

    assert cfg.llm.backend == "gguf"
    assert cfg.llm.api_key == ""
