from homie_core.voice.tts_melo import MeloTTS
from homie_core.voice.tts import BaseTTS


def test_melo_is_base_tts():
    tts = MeloTTS()
    assert isinstance(tts, BaseTTS)
    assert tts.name == "melo"


def test_melo_supports_indic():
    langs = MeloTTS().supported_languages
    assert "ta" in langs
    assert "te" in langs
    assert "ml" in langs


def test_melo_synthesize_without_model():
    assert MeloTTS().synthesize("hello") == b""


def test_melo_stream_without_model():
    chunks = list(MeloTTS().synthesize_stream("hello"))
    assert chunks == []


def test_melo_is_not_loaded_initially():
    assert MeloTTS().is_loaded is False


def test_melo_unload():
    tts = MeloTTS()
    tts.unload()
    assert tts.is_loaded is False
