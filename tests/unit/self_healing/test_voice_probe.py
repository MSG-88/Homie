# tests/unit/self_healing/test_voice_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.voice_probe import VoiceProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestVoiceProbe:
    def test_healthy_when_all_engines_available(self):
        vm = MagicMock()
        vm.available_engines = {"stt": True, "tts": True, "vad": True}
        vm.state = MagicMock(value="idle")
        probe = VoiceProbe(voice_manager=vm)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_tts_unavailable(self):
        vm = MagicMock()
        vm.available_engines = {"stt": True, "tts": False, "vad": True}
        vm.state = MagicMock(value="idle")
        probe = VoiceProbe(voice_manager=vm)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_stt_unavailable(self):
        vm = MagicMock()
        vm.available_engines = {"stt": False, "tts": False, "vad": False}
        vm.state = MagicMock(value="idle")
        probe = VoiceProbe(voice_manager=vm)
        result = probe.check()
        assert result.status == HealthStatus.FAILED

    def test_handles_voice_manager_none(self):
        probe = VoiceProbe(voice_manager=None)
        result = probe.check()
        assert result.status == HealthStatus.UNKNOWN
