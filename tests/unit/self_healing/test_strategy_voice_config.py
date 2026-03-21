# tests/unit/self_healing/test_strategy_voice_config.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.strategies.voice import (
    restart_voice_engine,
    switch_tts_engine,
    degrade_to_text_only,
)
from homie_core.self_healing.recovery.strategies.config import (
    reparse_config,
    use_last_known_good,
    reset_to_defaults,
)
from homie_core.self_healing.recovery.engine import RecoveryResult, RecoveryTier


class TestVoiceStrategies:
    def test_restart_voice_engine(self):
        vm = MagicMock()
        result = restart_voice_engine(module="voice", status=2, error="crash", voice_manager=vm)
        vm.stop.assert_called_once()
        vm.start.assert_called_once()
        assert result.tier == RecoveryTier.RETRY

    def test_switch_tts_engine(self):
        vm = MagicMock()
        result = switch_tts_engine(module="voice", status=2, error="tts fail", voice_manager=vm)
        assert result.tier == RecoveryTier.FALLBACK

    def test_degrade_to_text_only(self):
        vm = MagicMock()
        result = degrade_to_text_only(module="voice", status=2, error="fatal", voice_manager=vm)
        vm.stop.assert_called_once()
        assert result.success is True
        assert result.tier == RecoveryTier.DEGRADE


class TestConfigStrategies:
    def test_reparse_config(self, tmp_path):
        cfg_path = tmp_path / "homie.config.yaml"
        cfg_path.write_text("llm:\n  backend: gguf\n")
        result = reparse_config(module="config", status=2, error="parse error", config_path=str(cfg_path))
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_reparse_config_fails_on_missing(self, tmp_path):
        result = reparse_config(module="config", status=2, error="missing", config_path=str(tmp_path / "nope.yaml"))
        assert result.success is False

    def test_reset_to_defaults(self):
        config = MagicMock()
        result = reset_to_defaults(module="config", status=2, error="corrupt", config=config)
        assert result.tier == RecoveryTier.DEGRADE
