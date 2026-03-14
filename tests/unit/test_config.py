import os
import tempfile
from pathlib import Path

import pytest
import yaml

from homie_core.config import HomieConfig, load_config


@pytest.fixture
def tmp_config(tmp_path):
    cfg = {
        "llm": {"model_path": "models/test.gguf", "backend": "gguf"},
        "voice": {"enabled": False},
        "storage": {"path": str(tmp_path / ".homie")},
        "privacy": {"data_retention_days": 30},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def test_load_config_from_file(tmp_config):
    cfg = load_config(tmp_config)
    assert cfg.llm.backend == "gguf"
    assert cfg.voice.enabled is False


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.llm.backend == "gguf"
    assert cfg.storage.path is not None


def test_config_env_override(tmp_config, monkeypatch):
    monkeypatch.setenv("HOMIE_LLM_BACKEND", "transformers")
    cfg = load_config(tmp_config)
    assert cfg.llm.backend == "transformers"


def test_config_data_dir_created(tmp_path):
    cfg_data = {"storage": {"path": str(tmp_path / ".homie")}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = load_config(p)
    assert Path(cfg.storage.path).exists()


def test_inference_config_defaults():
    """InferenceConfig should have sensible defaults when not in YAML."""
    from homie_core.config import HomieConfig
    cfg = HomieConfig()
    assert cfg.inference.priority == ["local", "lan", "qubrid"]
    assert cfg.inference.qubrid.enabled is True
    assert cfg.inference.qubrid.model == "Qwen/Qwen3.5-Flash"
    assert cfg.inference.qubrid.base_url == "https://platform.qubrid.com/v1"
    assert cfg.inference.qubrid.timeout == 30
    assert cfg.inference.lan.prefer_desktop is True
    assert cfg.inference.lan.max_latency_ms == 500


def test_inference_config_from_yaml(tmp_path):
    """InferenceConfig should load from YAML."""
    from homie_core.config import load_config
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
inference:
  priority: [lan, local, qubrid]
  qubrid:
    enabled: false
    model: "custom/model"
    timeout: 60
  lan:
    prefer_desktop: false
    max_latency_ms: 1000
""")
    cfg = load_config(cfg_file)
    assert cfg.inference.priority == ["lan", "local", "qubrid"]
    assert cfg.inference.qubrid.enabled is False
    assert cfg.inference.qubrid.model == "custom/model"
    assert cfg.inference.qubrid.timeout == 60
    assert cfg.inference.lan.max_latency_ms == 1000
