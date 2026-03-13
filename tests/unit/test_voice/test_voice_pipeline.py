import queue
import struct
import threading
from unittest.mock import MagicMock
from homie_core.voice.voice_pipeline import VoicePipeline, PipelineState

def test_pipeline_states():
    assert PipelineState.IDLE.value == "idle"
    assert PipelineState.LISTENING.value == "listening"
    assert PipelineState.RECORDING.value == "recording"
    assert PipelineState.PROCESSING.value == "processing"
    assert PipelineState.SPEAKING.value == "speaking"

def test_pipeline_init():
    p = VoicePipeline(on_query=MagicMock(return_value=iter(["hi"])))
    assert p.state == PipelineState.IDLE

def test_barge_in_flushes():
    p = VoicePipeline(on_query=MagicMock(return_value=iter(["hi"])))
    p._playback_queue.put(b"a1")
    p._playback_queue.put(b"a2")
    assert p._playback_queue.qsize() == 2
    p.barge_in()
    assert p._playback_queue.empty()

def test_begin_listening():
    p = VoicePipeline()
    p.begin_listening()
    assert p.state == PipelineState.LISTENING

def test_stop_listening():
    p = VoicePipeline()
    p.begin_listening()
    p.stop_listening()
    assert p.state == PipelineState.IDLE

def test_recording_on_speech():
    p = VoicePipeline()
    p.vad = MagicMock()
    p.vad.is_speech.return_value = True
    p.begin_listening()
    p.process_audio_chunk(b"\x00" * 1024)
    assert p.state == PipelineState.RECORDING

def test_barge_in_during_speaking():
    p = VoicePipeline()
    p.vad = MagicMock()
    p.vad.is_speech.return_value = True
    p._set_state(PipelineState.SPEAKING)
    p._playback_queue.put(b"audio")
    p.process_audio_chunk(b"\x00" * 1024)
    assert p._playback_queue.empty()
    assert p.state == PipelineState.RECORDING

def test_mode_returns_idle_for_push_to_talk():
    callback = MagicMock(return_value=iter(["response."]))
    p = VoicePipeline(on_query=callback, mode="push_to_talk")
    p.vad = MagicMock()
    p.stt = MagicMock()
    p.stt.transcribe_bytes.return_value = ("test", "en")
    mock_engine = MagicMock()
    mock_engine.synthesize_stream.return_value = iter([b"audio"])
    p.tts_selector = MagicMock()
    p.tts_selector.select.return_value = mock_engine
    p._recording_buffer = [b"\x00" * 1024]
    p._silence_counter = 31
    p._set_state(PipelineState.RECORDING)
    p._process_recording()
    assert p.state == PipelineState.IDLE

def test_mode_returns_listening_for_conversational():
    callback = MagicMock(return_value=iter(["response."]))
    p = VoicePipeline(on_query=callback, mode="conversational")
    p.vad = MagicMock()
    p.stt = MagicMock()
    p.stt.transcribe_bytes.return_value = ("test", "en")
    mock_engine = MagicMock()
    mock_engine.synthesize_stream.return_value = iter([b"audio"])
    p.tts_selector = MagicMock()
    p.tts_selector.select.return_value = mock_engine
    p._recording_buffer = [b"\x00" * 1024]
    p._set_state(PipelineState.RECORDING)
    p._process_recording()
    assert p.state == PipelineState.LISTENING
