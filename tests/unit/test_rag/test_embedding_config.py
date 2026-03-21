from __future__ import annotations

import pytest


def test_embedding_config_defaults():
    from homie_core.config import EmbeddingConfig
    cfg = EmbeddingConfig()
    assert cfg.model_name == "BAAI/bge-base-en-v1.5"
    assert cfg.dimensions == 768
    assert cfg.batch_size == 32


def test_embedding_config_custom():
    from homie_core.config import EmbeddingConfig
    cfg = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", dimensions=384, batch_size=64)
    assert cfg.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.dimensions == 384
    assert cfg.batch_size == 64


def test_homie_config_has_embedding_field():
    from homie_core.config import HomieConfig
    cfg = HomieConfig()
    assert hasattr(cfg, "embedding")
    assert cfg.embedding.model_name == "BAAI/bge-base-en-v1.5"
    assert cfg.embedding.dimensions == 768
    assert cfg.embedding.batch_size == 32


def test_homie_config_embedding_from_yaml(tmp_path):
    from homie_core.config import load_config
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
embedding:
  model_name: "custom/model"
  dimensions: 512
  batch_size: 16
""")
    cfg = load_config(cfg_file)
    assert cfg.embedding.model_name == "custom/model"
    assert cfg.embedding.dimensions == 512
    assert cfg.embedding.batch_size == 16
