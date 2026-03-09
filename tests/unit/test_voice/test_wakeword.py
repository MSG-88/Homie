from unittest.mock import MagicMock, patch
from homie_core.voice.wakeword import WakeWordEngine


def test_init():
    engine = WakeWordEngine(wake_word="hey homie")
    assert engine.wake_word == "hey homie"
    assert engine.is_running is False


def test_stop():
    engine = WakeWordEngine()
    engine._running = True
    engine.stop()
    assert engine.is_running is False


def test_process_audio_when_not_running():
    engine = WakeWordEngine()
    result = engine.process_audio(b"\x00" * 1024)
    assert result is False
