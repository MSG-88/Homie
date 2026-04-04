"""Unit tests for the unified InferenceRouter.

Covers:
- Priority chain: local -> LAN -> Qubrid
- Fallback when local model not loaded
- Fallback when local engine raises
- All sources unavailable raises RuntimeError
- active_source property
- fallback_banner property
- stream() routing
- Tier-based timeouts (small/medium/large)
- Unknown / None tier handling
- Custom timeout preserved when no tier
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from homie_core.inference.router import InferenceRouter, _TIER_TIMEOUTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(priority=None, qubrid_enabled=True):
    cfg = MagicMock()
    cfg.inference.priority = priority or ["local", "lan", "qubrid"]
    cfg.inference.qubrid.enabled = qubrid_enabled
    cfg.inference.qubrid.model = "test-model"
    cfg.inference.qubrid.base_url = "https://api.example.com/v1"
    cfg.inference.qubrid.timeout = 30
    return cfg


def _make_router(
    priority=None,
    engine_loaded=False,
    qubrid_key="",
    qubrid_available=False,
    qubrid_enabled=True,
):
    engine = MagicMock()
    engine.is_loaded = engine_loaded
    engine.generate.return_value = "local response"
    engine.stream.return_value = iter(["local", " chunk"])

    cfg = _make_config(priority=priority, qubrid_enabled=qubrid_enabled)
    router = InferenceRouter(config=cfg, model_engine=engine, qubrid_api_key=qubrid_key)

    if qubrid_key and qubrid_available:
        qubrid_mock = MagicMock()
        qubrid_mock.is_available = True
        qubrid_mock.generate.return_value = "cloud response"
        qubrid_mock.stream.return_value = iter(["cloud", " chunk"])
        router._qubrid = qubrid_mock

    return router, engine


# ---------------------------------------------------------------------------
# _TIER_TIMEOUTS constants
# ---------------------------------------------------------------------------

class TestTierTimeoutsConstants:
    def test_small_timeout(self):
        assert _TIER_TIMEOUTS["small"] == 8

    def test_medium_timeout(self):
        assert _TIER_TIMEOUTS["medium"] == 25

    def test_large_timeout(self):
        assert _TIER_TIMEOUTS["large"] == 90

    def test_all_values_positive(self):
        for k, v in _TIER_TIMEOUTS.items():
            assert v > 0, f"{k} timeout must be positive"

    def test_ordered_small_to_large(self):
        assert _TIER_TIMEOUTS["small"] < _TIER_TIMEOUTS["medium"] < _TIER_TIMEOUTS["large"]


# ---------------------------------------------------------------------------
# generate() — local backend
# ---------------------------------------------------------------------------

class TestGenerateLocal:
    def test_uses_local_when_loaded(self):
        router, engine = _make_router(engine_loaded=True)
        result = router.generate("hello")
        assert result == "local response"
        engine.generate.assert_called_once()

    def test_local_receives_correct_params(self):
        router, engine = _make_router(engine_loaded=True)
        router.generate("q", max_tokens=512, temperature=0.3, stop=["<END>"])
        _, kwargs = engine.generate.call_args
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.3
        assert kwargs["stop"] == ["<END>"]

    def test_small_tier_sets_8s_timeout(self):
        router, engine = _make_router(engine_loaded=True)
        router.generate("q", tier="small")
        assert engine.generate.call_args[1]["timeout"] == 8

    def test_medium_tier_sets_25s_timeout(self):
        router, engine = _make_router(engine_loaded=True)
        router.generate("q", tier="medium")
        assert engine.generate.call_args[1]["timeout"] == 25

    def test_large_tier_sets_90s_timeout(self):
        router, engine = _make_router(engine_loaded=True)
        router.generate("q", tier="large")
        assert engine.generate.call_args[1]["timeout"] == 90

    def test_unknown_tier_uses_default_timeout(self):
        router, engine = _make_router(engine_loaded=True)
        router.generate("q", timeout=55, tier="xlarge")
        assert engine.generate.call_args[1]["timeout"] == 55

    def test_none_tier_does_not_override_timeout(self):
        router, engine = _make_router(engine_loaded=True)
        router.generate("q", timeout=42, tier=None)
        assert engine.generate.call_args[1]["timeout"] == 42

    def test_default_timeout_when_no_tier(self):
        router, engine = _make_router(engine_loaded=True)
        router.generate("q")
        assert engine.generate.call_args[1]["timeout"] == 120

    def test_model_hint_accepted(self):
        router, engine = _make_router(engine_loaded=True)
        # Should not raise even if engine ignores model hint
        result = router.generate("q", model="llama3")
        assert isinstance(result, str)

    def test_preferred_location_accepted(self):
        router, engine = _make_router(engine_loaded=True)
        result = router.generate("q", preferred_location="local")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# generate() — Qubrid fallback
# ---------------------------------------------------------------------------

class TestGenerateQubridFallback:
    def test_falls_back_to_qubrid_when_local_unavailable(self):
        router, _ = _make_router(engine_loaded=False, qubrid_key="key", qubrid_available=True)
        result = router.generate("hello")
        assert result == "cloud response"

    def test_qubrid_receives_correct_params(self):
        router, _ = _make_router(engine_loaded=False, qubrid_key="key", qubrid_available=True)
        router.generate("q", max_tokens=256, temperature=0.5)
        qubrid = router._qubrid
        _, kwargs = qubrid.generate.call_args
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.5

    def test_local_error_falls_through_to_qubrid(self):
        router, engine = _make_router(engine_loaded=True, qubrid_key="key", qubrid_available=True)
        engine.generate.side_effect = TimeoutError("timed out")
        result = router.generate("hello")
        assert result == "cloud response"

    def test_lan_source_is_skipped(self):
        """LAN source is configured but not implemented — should be silently skipped."""
        router, engine = _make_router(
            priority=["lan", "local"],
            engine_loaded=True,
        )
        result = router.generate("q")
        assert result == "local response"


# ---------------------------------------------------------------------------
# generate() — all sources fail
# ---------------------------------------------------------------------------

class TestGenerateAllFail:
    def test_raises_when_no_sources(self):
        router, _ = _make_router(engine_loaded=False)
        with pytest.raises(RuntimeError, match="All inference sources unavailable"):
            router.generate("hello")

    def test_error_message_contains_details(self):
        router, engine = _make_router(engine_loaded=True)
        engine.generate.side_effect = ValueError("model error")
        with pytest.raises(RuntimeError) as exc_info:
            router.generate("hello")
        assert "local" in str(exc_info.value).lower() or "unavailable" in str(exc_info.value).lower()

    def test_empty_priority_raises(self):
        router, _ = _make_router(priority=[], engine_loaded=False)
        with pytest.raises(RuntimeError):
            router.generate("hello")


# ---------------------------------------------------------------------------
# stream() routing
# ---------------------------------------------------------------------------

class TestStream:
    def test_stream_local_yields_chunks(self):
        router, engine = _make_router(engine_loaded=True)
        engine.stream.return_value = iter(["chunk1", "chunk2", "chunk3"])
        chunks = list(router.stream("hello"))
        assert chunks == ["chunk1", "chunk2", "chunk3"]
        engine.stream.assert_called_once()

    def test_stream_local_receives_params(self):
        router, engine = _make_router(engine_loaded=True)
        engine.stream.return_value = iter([])
        list(router.stream("q", max_tokens=100, temperature=0.2, stop=["STOP"]))
        _, kwargs = engine.stream.call_args
        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.2
        assert kwargs["stop"] == ["STOP"]

    def test_stream_falls_back_to_qubrid(self):
        router, _ = _make_router(engine_loaded=False, qubrid_key="key", qubrid_available=True)
        chunks = list(router.stream("hello"))
        assert chunks == ["cloud", " chunk"]

    def test_stream_local_error_falls_through(self):
        router, engine = _make_router(engine_loaded=True, qubrid_key="key", qubrid_available=True)
        engine.stream.side_effect = RuntimeError("stream failed")
        chunks = list(router.stream("hello"))
        assert chunks == ["cloud", " chunk"]

    def test_stream_all_fail_raises(self):
        router, _ = _make_router(engine_loaded=False)
        with pytest.raises(RuntimeError, match="All inference sources unavailable"):
            list(router.stream("hello"))

    def test_stream_tier_accepted(self):
        router, engine = _make_router(engine_loaded=True)
        engine.stream.return_value = iter(["ok"])
        list(router.stream("q", tier="small"))
        engine.stream.assert_called_once()

    def test_stream_model_hint_accepted(self):
        router, engine = _make_router(engine_loaded=True)
        engine.stream.return_value = iter(["ok"])
        result = list(router.stream("q", model="llama3"))
        assert result == ["ok"]


# ---------------------------------------------------------------------------
# active_source property
# ---------------------------------------------------------------------------

class TestActiveSource:
    def test_local_loaded_returns_local(self):
        router, _ = _make_router(engine_loaded=True)
        assert router.active_source == "Local"

    def test_local_not_loaded_qubrid_available_returns_cloud(self):
        router, _ = _make_router(engine_loaded=False, qubrid_key="key", qubrid_available=True)
        assert router.active_source == "Homie Intelligence (Cloud)"

    def test_no_sources_returns_none(self):
        router, _ = _make_router(engine_loaded=False)
        assert router.active_source == "None"

    def test_local_preferred_over_qubrid(self):
        router, _ = _make_router(engine_loaded=True, qubrid_key="key", qubrid_available=True)
        assert router.active_source == "Local"


# ---------------------------------------------------------------------------
# fallback_banner property
# ---------------------------------------------------------------------------

class TestFallbackBanner:
    def test_no_banner_when_local_loaded(self):
        router, _ = _make_router(engine_loaded=True)
        assert router.fallback_banner is None

    def test_banner_shown_when_cloud_active(self):
        router, _ = _make_router(engine_loaded=False, qubrid_key="key", qubrid_available=True)
        banner = router.fallback_banner
        assert banner is not None
        assert "local model" in banner.lower() or "No local model" in banner

    def test_no_banner_when_both_unavailable(self):
        router, _ = _make_router(engine_loaded=False)
        assert router.fallback_banner is None
