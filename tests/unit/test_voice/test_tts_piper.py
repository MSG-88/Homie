"""Tests for the standalone PiperTTS engine (tts_piper.py)."""
from unittest.mock import MagicMock, patch

import pytest

from homie_core.voice.tts_piper import PiperTTS
from homie_core.voice.tts import BaseTTS


# ---------------------------------------------------------------------------
# Instantiation & configuration
# ---------------------------------------------------------------------------

class TestPiperInit:
    def test_is_base_tts(self):
        assert isinstance(PiperTTS(), BaseTTS)

    def test_name(self):
        assert PiperTTS().name == "piper"

    def test_default_voice(self):
        assert PiperTTS().voice == "default"

    def test_custom_voice(self):
        tts = PiperTTS(voice="en_US-lessac-medium")
        assert tts.voice == "en_US-lessac-medium"

    def test_default_length_scale(self):
        assert PiperTTS().length_scale == 1.0

    def test_custom_length_scale(self):
        tts = PiperTTS(length_scale=0.8)
        assert tts.length_scale == 0.8

    def test_length_scale_clamped_low(self):
        tts = PiperTTS(length_scale=0.0)
        assert tts.length_scale == 0.1

    def test_length_scale_clamped_high(self):
        tts = PiperTTS(length_scale=5.0)
        assert tts.length_scale == 3.0

    def test_supported_languages(self):
        langs = PiperTTS().supported_languages
        assert "en" in langs
        assert "fr" in langs

    def test_not_loaded_initially(self):
        assert PiperTTS().is_loaded is False

    def test_repr(self):
        r = repr(PiperTTS(voice="test_voice", length_scale=1.2))
        assert "test_voice" in r
        assert "1.2" in r

    def test_speaker_id(self):
        tts = PiperTTS(speaker_id=3)
        assert tts._speaker_id == 3


# ---------------------------------------------------------------------------
# Behaviour without model
# ---------------------------------------------------------------------------

class TestPiperWithoutModel:
    def test_synthesize_returns_empty(self):
        assert PiperTTS().synthesize("hello") == b""

    def test_synthesize_stream_yields_nothing(self):
        assert list(PiperTTS().synthesize_stream("hello")) == []

    def test_unload_is_safe(self):
        tts = PiperTTS()
        tts.unload()
        assert tts.is_loaded is False

    def test_synthesize_to_file_returns_false(self, tmp_path):
        tts = PiperTTS()
        assert tts.synthesize_to_file("hello", tmp_path / "out.wav") is False


# ---------------------------------------------------------------------------
# Configuration changes
# ---------------------------------------------------------------------------

class TestPiperConfig:
    def test_set_length_scale(self):
        tts = PiperTTS()
        tts.set_length_scale(0.7)
        assert tts.length_scale == 0.7

    def test_set_length_scale_clamped(self):
        tts = PiperTTS()
        tts.set_length_scale(-1.0)
        assert tts.length_scale == 0.1
        tts.set_length_scale(99.0)
        assert tts.length_scale == 3.0


# ---------------------------------------------------------------------------
# Mocked model tests
# ---------------------------------------------------------------------------

class TestPiperWithMockedModel:
    @staticmethod
    def _make_loaded_tts(**kwargs):
        tts = PiperTTS(**kwargs)
        mock_model = MagicMock()
        tts._model = mock_model
        return tts, mock_model

    def test_synthesize_with_model(self):
        tts, mock_model = self._make_loaded_tts()
        mock_model.synthesize_stream_raw.return_value = [
            b"\x00" * 100, b"\x01" * 50
        ]

        result = tts.synthesize("hello world")
        assert len(result) == 150
        assert tts.is_loaded is True

    def test_synthesize_stream_with_model(self):
        tts, mock_model = self._make_loaded_tts()
        mock_model.synthesize_stream_raw.return_value = [
            b"\x00" * 100, b"\x01" * 50
        ]

        chunks = list(tts.synthesize_stream("hello"))
        assert len(chunks) == 2

    def test_synthesize_empty_text(self):
        tts, mock_model = self._make_loaded_tts()
        assert tts.synthesize("") == b""
        assert tts.synthesize("   ") == b""
        mock_model.synthesize_stream_raw.assert_not_called()

    def test_synthesize_with_speaker_id(self):
        tts, mock_model = self._make_loaded_tts(speaker_id=2)
        mock_model.synthesize_stream_raw.return_value = [b"\x00" * 50]

        tts.synthesize("hello")
        mock_model.synthesize_stream_raw.assert_called_once_with(
            "hello", speaker_id=2
        )

    def test_synthesize_with_length_scale(self):
        tts, mock_model = self._make_loaded_tts(length_scale=0.8)
        mock_model.synthesize_stream_raw.return_value = [b"\x00" * 50]

        tts.synthesize("hello")
        mock_model.synthesize_stream_raw.assert_called_once_with(
            "hello", length_scale=0.8
        )

    def test_synthesize_exception_returns_empty(self):
        tts, mock_model = self._make_loaded_tts()
        mock_model.synthesize_stream_raw.side_effect = RuntimeError("fail")

        assert tts.synthesize("hello") == b""

    def test_synthesize_stream_exception_stops(self):
        tts, mock_model = self._make_loaded_tts()
        mock_model.synthesize_stream_raw.side_effect = RuntimeError("fail")

        chunks = list(tts.synthesize_stream("hello"))
        assert chunks == []

    def test_synthesize_to_file(self, tmp_path):
        tts, mock_model = self._make_loaded_tts()
        mock_model.synthesize_stream_raw.return_value = [b"\xAB" * 100]

        out = tmp_path / "output.wav"
        assert tts.synthesize_to_file("hello", out) is True
        assert out.read_bytes() == b"\xAB" * 100

    def test_unload_clears_model(self):
        tts, _ = self._make_loaded_tts()
        tts.unload()
        assert tts.is_loaded is False


# ---------------------------------------------------------------------------
# Load behaviour
# ---------------------------------------------------------------------------

class TestPiperLoad:
    def test_load_without_piper(self, caplog):
        tts = PiperTTS()
        with patch("homie_core.voice.tts_piper._HAS_PIPER", False):
            with caplog.at_level("WARNING"):
                tts.load()
        assert tts.is_loaded is False

    def test_load_with_mock_piper(self):
        tts = PiperTTS(voice="test_model.onnx")
        mock_piper = MagicMock()
        with patch("homie_core.voice.tts_piper._HAS_PIPER", True), \
             patch("homie_core.voice.tts_piper._piper", mock_piper):
            tts.load()
        assert tts.is_loaded is True
        mock_piper.PiperVoice.load.assert_called_once_with("test_model.onnx")

    def test_load_file_not_found(self, caplog):
        tts = PiperTTS(voice="/nonexistent/model.onnx")
        mock_piper = MagicMock()
        mock_piper.PiperVoice.load.side_effect = FileNotFoundError("nope")
        with patch("homie_core.voice.tts_piper._HAS_PIPER", True), \
             patch("homie_core.voice.tts_piper._piper", mock_piper):
            tts.load()
        assert tts.is_loaded is False

    def test_load_exception_handled(self):
        tts = PiperTTS()
        mock_piper = MagicMock()
        mock_piper.PiperVoice.load.side_effect = RuntimeError("bad model")
        with patch("homie_core.voice.tts_piper._HAS_PIPER", True), \
             patch("homie_core.voice.tts_piper._piper", mock_piper):
            tts.load()
        assert tts.is_loaded is False
