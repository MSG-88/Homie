"""Tests for TTS base class and the legacy PiperTTS in tts.py."""
import pytest
from unittest.mock import MagicMock, patch

from homie_core.voice.tts import BaseTTS, PiperTTS, TextToSpeech


# ---------------------------------------------------------------------------
# BaseTTS abstract contract
# ---------------------------------------------------------------------------

class TestBaseTTS:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseTTS()

    def test_default_is_loaded_is_false(self):
        """The base class default is_loaded returns False."""
        # Access via a concrete subclass that doesn't override
        class Stub(BaseTTS):
            def load(self, device="cpu"): ...
            def synthesize(self, text): return b""
            def synthesize_stream(self, text): yield b""
            def unload(self): ...
            @property
            def supported_languages(self): return []
            @property
            def name(self): return "stub"

        # Don't override is_loaded => defaults to False from BaseTTS
        s = Stub()
        # Stub inherits BaseTTS.is_loaded which always returns False
        assert s.is_loaded is False


# ---------------------------------------------------------------------------
# PiperTTS (in tts.py — the original implementation)
# ---------------------------------------------------------------------------

class TestPiperTTSOriginal:
    def test_name(self):
        assert PiperTTS().name == "piper"

    def test_supported_languages(self):
        langs = PiperTTS().supported_languages
        assert "en" in langs
        assert "fr" in langs

    def test_not_loaded_initially(self):
        assert PiperTTS().is_loaded is False

    def test_synthesize_without_model(self):
        assert PiperTTS().synthesize("hello") == b""

    def test_synthesize_stream_without_model(self):
        chunks = list(PiperTTS().synthesize_stream("hello"))
        assert chunks == []

    def test_unload(self):
        tts = PiperTTS()
        tts.unload()
        assert tts.is_loaded is False

    def test_synthesize_to_file_without_model(self, tmp_path):
        tts = PiperTTS()
        result = tts.synthesize_to_file("hello", tmp_path / "out.wav")
        assert result is False

    def test_synthesize_with_mock_model(self):
        tts = PiperTTS()
        mock = MagicMock()
        mock.synthesize_stream_raw.return_value = [b"\x00" * 100, b"\x01" * 50]
        tts._model = mock

        result = tts.synthesize("hello world")
        assert len(result) == 150
        assert tts.is_loaded is True

    def test_synthesize_stream_with_mock_model(self):
        tts = PiperTTS()
        mock = MagicMock()
        mock.synthesize_stream_raw.return_value = [b"\x00" * 100, b"\x01" * 50]
        tts._model = mock

        chunks = list(tts.synthesize_stream("hello"))
        assert len(chunks) == 2

    def test_synthesize_exception_returns_empty(self):
        tts = PiperTTS()
        mock = MagicMock()
        mock.synthesize_stream_raw.side_effect = RuntimeError("fail")
        tts._model = mock

        assert tts.synthesize("hello") == b""

    def test_load_without_piper(self, caplog):
        tts = PiperTTS()
        with patch.dict("sys.modules", {"piper": None}):
            tts.load()
        assert tts.is_loaded is False


# ---------------------------------------------------------------------------
# TextToSpeech legacy alias
# ---------------------------------------------------------------------------

class TestTextToSpeechAlias:
    def test_is_subclass_of_piper(self):
        assert issubclass(TextToSpeech, PiperTTS)

    def test_voice_attribute(self):
        tts = TextToSpeech(voice="custom_voice")
        assert tts.voice == "custom_voice"

    def test_inherits_behaviour(self):
        tts = TextToSpeech()
        assert tts.synthesize("test") == b""
        assert tts.is_loaded is False
