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


def test_load_config_from_file(tmp_config, monkeypatch):
    monkeypatch.delenv("HF_KEY", raising=False)
    cfg = load_config(tmp_config)
    assert cfg.llm.backend == "gguf"
    assert cfg.voice.enabled is False


def test_load_config_defaults(monkeypatch):
    monkeypatch.delenv("HF_KEY", raising=False)
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


def test_email_config_defaults():
    from homie_core.config import HomieConfig
    cfg = HomieConfig()
    assert cfg.email.auto_download_attachments is True
    assert cfg.email.max_attachment_size_mb == 25
    assert cfg.email.knowledge_extraction is True
    assert cfg.email.extraction_batch_size == 20
    assert cfg.email.send_requires_confirmation is True
    assert cfg.email.insight_refresh_interval == 3600
    assert cfg.email.auto_download_categories == ["bill", "order", "work"]


def test_config_has_mesh_section():
    cfg = HomieConfig()
    assert hasattr(cfg, "mesh")
    assert cfg.mesh.enabled is True
    assert cfg.mesh.auto_discover is True
    assert cfg.mesh.auto_elect_hub is True
    assert cfg.mesh.preferred_role == "auto"
    assert cfg.mesh.heartbeat_interval == 15
    assert cfg.mesh.sync_interval == 30


def test_mesh_config_from_yaml(tmp_path):
    data = {
        "mesh": {
            "enabled": False,
            "preferred_role": "hub",
            "heartbeat_interval": 10,
            "wan": {"enabled": True, "transport": "websocket"},
            "inference": {"max_concurrent": 4},
            "security": {"key_rotation_days": 7},
        },
        "storage": {"path": str(tmp_path / ".homie")},
    }
    p = tmp_path / "config.yaml"
    import yaml
    p.write_text(yaml.dump(data))
    cfg = load_config(p)
    assert cfg.mesh.enabled is False
    assert cfg.mesh.preferred_role == "hub"
    assert cfg.mesh.heartbeat_interval == 10
    assert cfg.mesh.wan.enabled is True
    assert cfg.mesh.wan.transport == "websocket"
    assert cfg.mesh.inference.max_concurrent == 4
    assert cfg.mesh.security.key_rotation_days == 7
