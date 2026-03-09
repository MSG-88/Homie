from homie_core.voice.audio_io import AudioIO


def test_audio_io_init():
    aio = AudioIO(sample_rate=16000, channels=1)
    assert aio.sample_rate == 16000
    assert aio.channels == 1
    assert aio.is_recording is False
    assert aio.is_playing is False


def test_stop_recording_without_start():
    aio = AudioIO()
    aio.stop_recording()  # should not raise
    assert aio.is_recording is False


def test_stop_playback_without_play():
    aio = AudioIO()
    aio.stop_playback()  # should not raise
    assert aio.is_playing is False
