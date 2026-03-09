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
