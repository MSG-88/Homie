from __future__ import annotations

import pytest

from homie_core.config import ContextConfig, HomieConfig


def test_context_config_defaults():
    cfg = ContextConfig()
    assert cfg.summarize_trigger_pct == 0.85
    assert cfg.summarize_keep_pct == 0.10
    assert cfg.manual_compact_pct == 0.50
    assert cfg.large_result_threshold == 80000
    assert cfg.long_line_threshold == 5000
    assert cfg.arg_truncation_threshold == 2000


def test_context_config_custom_values():
    cfg = ContextConfig(
        summarize_trigger_pct=0.9,
        summarize_keep_pct=0.2,
        manual_compact_pct=0.6,
        large_result_threshold=50000,
        long_line_threshold=3000,
        arg_truncation_threshold=1000,
    )
    assert cfg.summarize_trigger_pct == 0.9
    assert cfg.summarize_keep_pct == 0.2
    assert cfg.manual_compact_pct == 0.6
    assert cfg.large_result_threshold == 50000
    assert cfg.long_line_threshold == 3000
    assert cfg.arg_truncation_threshold == 1000


def test_homie_config_has_context_field():
    cfg = HomieConfig()
    assert hasattr(cfg, "context")
    assert isinstance(cfg.context, ContextConfig)


def test_homie_config_context_uses_defaults():
    cfg = HomieConfig()
    assert cfg.context.summarize_trigger_pct == 0.85
    assert cfg.context.large_result_threshold == 80000
    assert cfg.context.long_line_threshold == 5000
    assert cfg.context.arg_truncation_threshold == 2000


def test_homie_config_context_can_be_overridden():
    cfg = HomieConfig(context=ContextConfig(large_result_threshold=10000))
    assert cfg.context.large_result_threshold == 10000
    # Other fields still default
    assert cfg.context.long_line_threshold == 5000
