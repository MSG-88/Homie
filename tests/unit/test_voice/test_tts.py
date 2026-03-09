from homie_core.voice.tts import TextToSpeech


def test_init():
    tts = TextToSpeech(voice="default")
    assert tts.voice == "default"
    assert tts.is_loaded is False


def test_synthesize_without_model():
    tts = TextToSpeech()
    result = tts.synthesize("Hello world")
    assert result == b""


def test_unload():
    tts = TextToSpeech()
    tts.unload()
    assert tts.is_loaded is False
