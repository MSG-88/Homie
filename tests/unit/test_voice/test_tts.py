import pytest
from homie_core.voice.tts import BaseTTS, PiperTTS


def test_base_tts_is_abstract():
    with pytest.raises(TypeError):
        BaseTTS()


def test_piper_implements_base():
    tts = PiperTTS(voice="default")
    assert tts.name == "piper"
    assert "en" in tts.supported_languages


def test_piper_synthesize_without_model():
    tts = PiperTTS()
    assert tts.synthesize("hello") == b""


def test_piper_is_not_loaded_initially():
    tts = PiperTTS()
    assert tts.is_loaded is False


def test_piper_unload():
    tts = PiperTTS()
    tts.unload()
    assert tts.is_loaded is False


def test_piper_synthesize_stream_without_model():
    tts = PiperTTS()
    chunks = list(tts.synthesize_stream("hello"))
    assert chunks == []
