import struct
from homie_core.voice.vad import VoiceActivityDetector


def test_init():
    vad = VoiceActivityDetector(energy_threshold=500)
    assert vad.is_speaking is False


def test_silence_detection():
    vad = VoiceActivityDetector(energy_threshold=500, silence_frames=3)
    silent_chunk = b"\x00" * 2048
    for _ in range(5):
        vad.process(silent_chunk)
    assert vad.is_speaking is False


def test_speech_detection():
    vad = VoiceActivityDetector(energy_threshold=100)
    # Create loud audio chunk
    loud_chunk = struct.pack("<" + "h" * 1024, *([10000] * 1024))
    result = vad.process(loud_chunk)
    assert result is True
    assert vad.is_speaking is True


def test_reset():
    vad = VoiceActivityDetector()
    vad._is_speaking = True
    vad.reset()
    assert vad.is_speaking is False


def test_average_energy():
    vad = VoiceActivityDetector()
    assert vad.get_average_energy() == 0.0
