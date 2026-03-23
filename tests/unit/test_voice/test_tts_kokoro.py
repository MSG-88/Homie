"""Tests for the KokoroTTS engine."""
from unittest.mock import MagicMock, patch

import pytest

from homie_core.voice.tts_kokoro import KokoroTTS
from homie_core.voice.tts import BaseTTS


# ---------------------------------------------------------------------------
# Instantiation & configuration
# ---------------------------------------------------------------------------

class TestKokoroInit:
    def test_is_base_tts(self):
        assert isinstance(KokoroTTS(), BaseTTS)

    def test_name(self):
        assert KokoroTTS().name == "kokoro"

    def test_default_voice(self):
        assert KokoroTTS().voice == "af_heart"

    def test_custom_voice(self):
        tts = KokoroTTS(voice="am_adam")
        assert tts.voice == "am_adam"

    def test_default_speed(self):
        assert KokoroTTS().speed == 1.0

    def test_custom_speed(self):
        tts = KokoroTTS(speed=1.5)
        assert tts.speed == 1.5

    def test_speed_clamped_low(self):
        tts = KokoroTTS(speed=0.0)
        assert tts.speed == 0.1

    def test_speed_clamped_high(self):
        tts = KokoroTTS(speed=10.0)
        assert tts.speed == 3.0

    def test_supported_languages(self):
        langs = KokoroTTS().supported_languages
        assert "en" in langs
        assert "fr" in langs
        assert "ja" in langs
        assert "zh" in langs

    def test_not_loaded_initially(self):
        assert KokoroTTS().is_loaded is False

    def test_repr(self):
        r = repr(KokoroTTS(voice="af_heart", lang="en", speed=1.2))
        assert "af_heart" in r
        assert "1.2" in r


# ---------------------------------------------------------------------------
# Behaviour without pipeline
# ---------------------------------------------------------------------------

class TestKokoroWithoutPipeline:
    def test_synthesize_returns_empty(self):
        assert KokoroTTS().synthesize("hello") == b""

    def test_synthesize_stream_yields_nothing(self):
        assert list(KokoroTTS().synthesize_stream("hello")) == []

    def test_unload_is_safe(self):
        tts = KokoroTTS()
        tts.unload()
        assert tts.is_loaded is False


# ---------------------------------------------------------------------------
# Configuration changes
# ---------------------------------------------------------------------------

class TestKokoroConfig:
    def test_set_voice(self):
        tts = KokoroTTS()
        tts.set_voice("af_bella")
        assert tts.voice == "af_bella"

    def test_set_speed(self):
        tts = KokoroTTS()
        tts.set_speed(0.8)
        assert tts.speed == 0.8

    def test_set_speed_clamped(self):
        tts = KokoroTTS()
        tts.set_speed(-1.0)
        assert tts.speed == 0.1
        tts.set_speed(99.0)
        assert tts.speed == 3.0


# ---------------------------------------------------------------------------
# Mocked pipeline tests
# ---------------------------------------------------------------------------

class TestKokoroWithMockedPipeline:
    @staticmethod
    def _make_loaded_tts():
        tts = KokoroTTS()
        mock_pipeline = MagicMock()
        tts._pipeline = mock_pipeline
        return tts, mock_pipeline

    def test_synthesize_with_pipeline(self):
        tts, mock_pipeline = self._make_loaded_tts()
        # Simulate pipeline yielding numpy-like arrays
        import numpy as np
        audio_chunk = np.zeros(1000, dtype=np.float32)
        mock_pipeline.return_value = [("gs", "ps", audio_chunk)]

        result = tts.synthesize("hello world")
        assert isinstance(result, bytes)
        assert len(result) > 0
        mock_pipeline.assert_called_once_with(
            "hello world", voice="af_heart", speed=1.0
        )

    def test_synthesize_stream_with_pipeline(self):
        tts, mock_pipeline = self._make_loaded_tts()
        import numpy as np
        chunk1 = np.ones(500, dtype=np.float32) * 0.5
        chunk2 = np.ones(300, dtype=np.float32) * 0.3
        mock_pipeline.return_value = [
            ("g1", "p1", chunk1),
            ("g2", "p2", chunk2),
        ]

        chunks = list(tts.synthesize_stream("hello world"))
        assert len(chunks) == 2
        assert all(isinstance(c, bytes) for c in chunks)

    def test_synthesize_empty_text(self):
        tts, mock_pipeline = self._make_loaded_tts()
        assert tts.synthesize("") == b""
        assert tts.synthesize("   ") == b""
        mock_pipeline.assert_not_called()

    def test_synthesize_exception_returns_empty(self):
        tts, mock_pipeline = self._make_loaded_tts()
        mock_pipeline.side_effect = RuntimeError("synthesis error")

        assert tts.synthesize("hello") == b""

    def test_synthesize_stream_exception_stops(self):
        tts, mock_pipeline = self._make_loaded_tts()
        mock_pipeline.side_effect = RuntimeError("stream error")

        chunks = list(tts.synthesize_stream("hello"))
        assert chunks == []

    def test_is_loaded_with_pipeline(self):
        tts, _ = self._make_loaded_tts()
        assert tts.is_loaded is True

    def test_unload_clears_pipeline(self):
        tts, _ = self._make_loaded_tts()
        tts.unload()
        assert tts.is_loaded is False


# ---------------------------------------------------------------------------
# Load behaviour
# ---------------------------------------------------------------------------

class TestKokoroLoad:
    def test_load_without_kokoro(self, caplog):
        tts = KokoroTTS()
        with patch("homie_core.voice.tts_kokoro._HAS_KOKORO", False):
            with caplog.at_level("WARNING"):
                tts.load()
        assert tts.is_loaded is False

    def test_load_with_mock_kokoro(self):
        tts = KokoroTTS(lang="en")
        mock_cls = MagicMock()
        with patch("homie_core.voice.tts_kokoro._HAS_KOKORO", True), \
             patch("homie_core.voice.tts_kokoro._KPipeline", mock_cls):
            tts.load(device="cpu")
        assert tts.is_loaded is True
        mock_cls.assert_called_once_with(lang_code="a", device="cpu")

    def test_load_exception_handled(self):
        tts = KokoroTTS()
        mock_cls = MagicMock(side_effect=RuntimeError("GPU fail"))
        with patch("homie_core.voice.tts_kokoro._HAS_KOKORO", True), \
             patch("homie_core.voice.tts_kokoro._KPipeline", mock_cls):
            tts.load()
        assert tts.is_loaded is False
