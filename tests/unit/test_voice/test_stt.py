from homie_core.voice.stt import SpeechToText


def test_init():
    stt = SpeechToText(model_size="tiny")
    assert stt.model_size == "tiny"
    assert stt.is_loaded is False


def test_transcribe_without_model():
    stt = SpeechToText()
    result = stt.transcribe("nonexistent.wav")
    assert result == ""


def test_transcribe_bytes_without_model():
    stt = SpeechToText()
    result = stt.transcribe_bytes(b"\x00" * 1024)
    assert result == ""


def test_unload():
    stt = SpeechToText()
    stt.unload()
    assert stt.is_loaded is False
