"""Tests for the SpeechToText engine (faster-whisper backend)."""
import sys
from unittest.mock import MagicMock, patch

import pytest

from homie_core.voice.stt import SpeechToText, SUPPORTED_MODELS


# ---------------------------------------------------------------------------
# Instantiation & configuration
# ---------------------------------------------------------------------------

class TestSTTInit:
    def test_default_model_size(self):
        stt = SpeechToText()
        assert stt.model_size == "base.en"

    def test_custom_model_size(self):
        for size in ("tiny.en", "base.en", "small", "medium", "large-v3"):
            stt = SpeechToText(model_size=size)
            assert stt.model_size == size

    def test_custom_device_and_compute(self):
        stt = SpeechToText(device="cuda", compute_type="float16")
        assert stt.device == "cuda"
        assert stt.compute_type == "float16"

    def test_beam_size(self):
        stt = SpeechToText(beam_size=3)
        assert stt.beam_size == 3

    def test_forced_language(self):
        stt = SpeechToText(language="fr")
        assert stt.language == "fr"

    def test_not_loaded_initially(self):
        stt = SpeechToText()
        assert stt.is_loaded is False

    def test_repr(self):
        stt = SpeechToText(model_size="tiny.en")
        r = repr(stt)
        assert "tiny.en" in r
        assert "not loaded" in r

    def test_unknown_model_size_warns(self, caplog):
        """Unknown model names should log a warning but not raise."""
        with caplog.at_level("WARNING"):
            stt = SpeechToText(model_size="nonexistent-model")
        assert stt.model_size == "nonexistent-model"

    def test_supported_models_constant(self):
        assert "tiny.en" in SUPPORTED_MODELS
        assert "base.en" in SUPPORTED_MODELS
        assert "large-v3" in SUPPORTED_MODELS


# ---------------------------------------------------------------------------
# Behaviour without a loaded model
# ---------------------------------------------------------------------------

class TestSTTWithoutModel:
    def test_transcribe_returns_empty(self):
        stt = SpeechToText()
        text, lang = stt.transcribe("nonexistent.wav")
        assert text == ""
        assert lang == "en"

    def test_transcribe_bytes_returns_empty(self):
        stt = SpeechToText()
        text, lang = stt.transcribe_bytes(b"\x00" * 3200)
        assert text == ""
        assert lang == "en"

    def test_unload_is_safe(self):
        stt = SpeechToText()
        stt.unload()
        assert stt.is_loaded is False


# ---------------------------------------------------------------------------
# Behaviour with a mocked model
# ---------------------------------------------------------------------------

class TestSTTWithMockedModel:
    @staticmethod
    def _make_stt_with_mock():
        stt = SpeechToText(model_size="base.en")
        mock_model = MagicMock()
        stt._model = mock_model
        return stt, mock_model

    def test_transcribe_file(self):
        stt, mock_model = self._make_stt_with_mock()
        seg = MagicMock()
        seg.text = "hello world"
        info = MagicMock()
        info.language = "en"
        mock_model.transcribe.return_value = ([seg], info)

        text, lang = stt.transcribe("test.wav")
        assert text == "hello world"
        assert lang == "en"
        mock_model.transcribe.assert_called_once()

    def test_transcribe_bytes(self):
        stt, mock_model = self._make_stt_with_mock()
        seg = MagicMock()
        seg.text = "testing bytes"
        info = MagicMock()
        info.language = "en"
        mock_model.transcribe.return_value = ([seg], info)

        text, lang = stt.transcribe_bytes(b"\x00" * 3200, sample_rate=16000)
        assert text == "testing bytes"
        assert lang == "en"

    def test_transcribe_multiple_segments(self):
        stt, mock_model = self._make_stt_with_mock()
        seg1 = MagicMock()
        seg1.text = "hello"
        seg2 = MagicMock()
        seg2.text = "world"
        info = MagicMock()
        info.language = "fr"
        mock_model.transcribe.return_value = ([seg1, seg2], info)

        text, lang = stt.transcribe("test.wav")
        assert text == "hello world"
        assert lang == "fr"

    def test_transcribe_empty_audio_bytes(self):
        stt, mock_model = self._make_stt_with_mock()
        # Empty bytes => zero-length array => should return empty
        text, lang = stt.transcribe_bytes(b"")
        assert text == ""
        assert lang == "en"

    def test_transcribe_exception_returns_empty(self):
        stt, mock_model = self._make_stt_with_mock()
        mock_model.transcribe.side_effect = RuntimeError("boom")

        text, lang = stt.transcribe("test.wav")
        assert text == ""
        assert lang == "en"

    def test_transcribe_bytes_exception_returns_empty(self):
        stt, mock_model = self._make_stt_with_mock()
        mock_model.transcribe.side_effect = RuntimeError("boom")

        text, lang = stt.transcribe_bytes(b"\x00" * 3200)
        assert text == ""
        assert lang == "en"

    def test_is_loaded_with_model(self):
        stt, _ = self._make_stt_with_mock()
        assert stt.is_loaded is True

    def test_unload_clears_model(self):
        stt, _ = self._make_stt_with_mock()
        stt.unload()
        assert stt.is_loaded is False


# ---------------------------------------------------------------------------
# Load behaviour
# ---------------------------------------------------------------------------

class TestSTTLoad:
    def test_load_without_faster_whisper(self, caplog):
        """When faster-whisper is not installed, load() logs a warning."""
        stt = SpeechToText()
        with patch("homie_core.voice.stt._HAS_FASTER_WHISPER", False):
            with caplog.at_level("WARNING"):
                stt.load()
        assert stt.is_loaded is False

    def test_load_with_mock_faster_whisper(self):
        """When faster-whisper is available, load() creates the model."""
        stt = SpeechToText(model_size="tiny.en")
        mock_cls = MagicMock()
        with patch("homie_core.voice.stt._HAS_FASTER_WHISPER", True), \
             patch("homie_core.voice.stt._WhisperModel", mock_cls):
            stt.load()
        assert stt.is_loaded is True
        mock_cls.assert_called_once_with(
            "tiny.en", device="cpu", compute_type="float32"
        )

    def test_load_exception_handled(self, caplog):
        """If model init fails, load() catches the exception."""
        stt = SpeechToText()
        mock_cls = MagicMock(side_effect=RuntimeError("GPU OOM"))
        with patch("homie_core.voice.stt._HAS_FASTER_WHISPER", True), \
             patch("homie_core.voice.stt._WhisperModel", mock_cls):
            stt.load()
        assert stt.is_loaded is False
