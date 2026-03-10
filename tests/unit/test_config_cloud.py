"""Tests for cloud-related LLMConfig fields (api_key, api_base_url)."""
from __future__ import annotations

import yaml

from homie_core.config import HomieConfig, LLMConfig, load_config


def test_llm_config_has_cloud_fields():
    cfg = LLMConfig()
    assert cfg.api_key == ""
    assert cfg.api_base_url == ""


def test_cloud_config_roundtrip():
    cfg = HomieConfig(
        llm=LLMConfig(
            backend="cloud",
            model_path="gpt-4o",
            api_key="sk-test-123",
            api_base_url="https://api.openai.com/v1",
        )
    )
    data = cfg.model_dump()
    restored = HomieConfig(**data)
    assert restored.llm.backend == "cloud"
    assert restored.llm.model_path == "gpt-4o"
    assert restored.llm.api_key == "sk-test-123"
    assert restored.llm.api_base_url == "https://api.openai.com/v1"


def test_cloud_config_yaml_roundtrip(tmp_path):
    cfg = HomieConfig(
        llm=LLMConfig(
            backend="cloud",
            model_path="gpt-4o",
            api_key="sk-test-123",
            api_base_url="https://api.openai.com/v1",
        )
    )
    yaml_file = tmp_path / "homie.config.yaml"
    yaml_file.write_text(yaml.dump(cfg.model_dump()))

    loaded = load_config(yaml_file)
    assert loaded.llm.backend == "cloud"
    assert loaded.llm.model_path == "gpt-4o"
    assert loaded.llm.api_key == "sk-test-123"
    assert loaded.llm.api_base_url == "https://api.openai.com/v1"


def test_api_key_env_override(tmp_path, monkeypatch):
    cfg = HomieConfig(
        llm=LLMConfig(
            backend="cloud",
            api_key="from-config",
        )
    )
    yaml_file = tmp_path / "homie.config.yaml"
    yaml_file.write_text(yaml.dump(cfg.model_dump()))

    monkeypatch.setenv("HOMIE_API_KEY", "from-env")
    loaded = load_config(yaml_file)
    assert loaded.llm.api_key == "from-env"
