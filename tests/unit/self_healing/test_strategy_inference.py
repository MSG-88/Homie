# tests/unit/self_healing/test_strategy_inference.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.recovery.strategies.inference import (
    retry_with_reduced_tokens,
    fallback_reduce_context,
    switch_to_smaller_model,
    degrade_to_cached,
)
from homie_core.self_healing.recovery.engine import RecoveryResult, RecoveryTier


class TestInferenceRecoveryStrategies:
    def test_retry_reduced_tokens_success(self):
        engine = MagicMock()
        engine.generate.return_value = "ok"
        result = retry_with_reduced_tokens(module="inference", status=2, error="timeout", model_engine=engine)
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_retry_reduced_tokens_failure(self):
        engine = MagicMock()
        engine.generate.side_effect = TimeoutError("still slow")
        result = retry_with_reduced_tokens(module="inference", status=2, error="timeout", model_engine=engine)
        assert result.success is False

    def test_fallback_reduce_context(self):
        config = MagicMock()
        config.llm.context_length = 65536
        engine = MagicMock()
        engine.generate.return_value = "ok"
        result = fallback_reduce_context(module="inference", status=2, error="oom", config=config, model_engine=engine)
        assert result.success is True
        assert result.tier == RecoveryTier.FALLBACK

    def test_switch_to_smaller_model(self):
        engine = MagicMock()
        registry = MagicMock()
        registry.list_models.return_value = [MagicMock(name="small-model")]
        result = switch_to_smaller_model(module="inference", status=2, error="oom", model_engine=engine, model_registry=registry)
        assert result.tier == RecoveryTier.REBUILD

    def test_degrade_to_cached(self):
        result = degrade_to_cached(module="inference", status=2, error="fatal")
        assert result.success is True
        assert result.tier == RecoveryTier.DEGRADE
