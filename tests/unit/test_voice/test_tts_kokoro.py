from homie_core.voice.tts_kokoro import KokoroTTS
from homie_core.voice.tts import BaseTTS


def test_kokoro_is_base_tts():
    tts = KokoroTTS()
    assert isinstance(tts, BaseTTS)
    assert tts.name == "kokoro"


def test_kokoro_supported_languages():
    assert "en" in KokoroTTS().supported_languages
    assert "fr" in KokoroTTS().supported_languages


def test_kokoro_synthesize_without_model():
    assert KokoroTTS().synthesize("hello") == b""


def test_kokoro_stream_without_model():
    chunks = list(KokoroTTS().synthesize_stream("hello"))
    assert chunks == []


def test_kokoro_is_not_loaded_initially():
    assert KokoroTTS().is_loaded is False


def test_kokoro_unload():
    tts = KokoroTTS()
    tts.unload()
    assert tts.is_loaded is False
