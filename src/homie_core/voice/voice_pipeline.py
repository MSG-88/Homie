from __future__ import annotations
import enum
import logging
import queue
import threading
import time
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

class PipelineState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"
    SPEAKING = "speaking"

class VoicePipeline:
    _SILENCE_CHUNKS_THRESHOLD = 30
    _MAX_RECORDING_CHUNKS = 500
    _QUEUE_MAX_DEPTH = 50

    def __init__(
        self,
        on_query: Callable[[str], Iterator[str]] | None = None,
        on_state_change: Callable[[PipelineState], None] | None = None,
        on_response_complete: Callable[[], None] | None = None,
        mode: str = "hybrid",
        sample_rate: int = 16000,
        chunk_size: int = 512,
    ) -> None:
        self._on_query = on_query
        self._on_state_change = on_state_change
        self._on_response_complete = on_response_complete
        self._mode = mode
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._should_listen = threading.Event()
        self._should_play = threading.Event()
        self._should_play.set()

        self._vad_queue: queue.Queue = queue.Queue()
        self._playback_queue: queue.Queue = queue.Queue()

        self.vad = None
        self.stt = None
        self.tts_selector = None
        self.audio_in = None
        self.audio_out = None
        self.wake_word = None

        self._threads: list[threading.Thread] = []
        self._recording_buffer: list[bytes] = []
        self._silence_counter = 0

    @property
    def state(self) -> PipelineState:
        return self._state

    def _set_state(self, new_state: PipelineState) -> None:
        with self._state_lock:
            if self._state != new_state:
                logger.debug("Pipeline: %s -> %s", self._state.value, new_state.value)
                self._state = new_state
                if self._on_state_change:
                    try:
                        self._on_state_change(new_state)
                    except Exception:
                        logger.exception("State change callback failed")

    def start(self) -> None:
        self._stop_event.clear()
        if self.audio_in:
            t = threading.Thread(target=self.audio_in.run, daemon=True, name="audio-in")
            t.start()
            self._threads.append(t)
        if self.audio_out:
            t = threading.Thread(target=self.audio_out.run, daemon=True, name="audio-out")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        self._vad_queue.put(b"END")
        self._playback_queue.put(b"END")
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        self._set_state(PipelineState.IDLE)

    def begin_listening(self) -> None:
        self._should_listen.set()
        self._set_state(PipelineState.LISTENING)
        self._recording_buffer.clear()
        self._silence_counter = 0

    def stop_listening(self) -> None:
        self._should_listen.clear()
        self._set_state(PipelineState.IDLE)

    def barge_in(self) -> None:
        self._should_play.clear()
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except queue.Empty:
                break
        self._should_play.set()
        logger.debug("Barge-in: TTS flushed")

    def process_audio_chunk(self, chunk: bytes) -> None:
        if not self.vad:
            return
        is_speech = self.vad.is_speech(chunk)

        if self._state == PipelineState.LISTENING:
            if is_speech:
                self._set_state(PipelineState.RECORDING)
                self._recording_buffer = [chunk]
                self._silence_counter = 0
        elif self._state == PipelineState.RECORDING:
            self._recording_buffer.append(chunk)
            if not is_speech:
                self._silence_counter += 1
            else:
                self._silence_counter = 0
            if (self._silence_counter >= self._SILENCE_CHUNKS_THRESHOLD
                    or len(self._recording_buffer) >= self._MAX_RECORDING_CHUNKS):
                self._process_recording()
        elif self._state == PipelineState.SPEAKING:
            if is_speech:
                self.barge_in()
                self._set_state(PipelineState.RECORDING)
                self._recording_buffer = [chunk]
                self._silence_counter = 0

    def _process_recording(self) -> None:
        self._set_state(PipelineState.PROCESSING)
        audio_bytes = b"".join(self._recording_buffer)
        self._recording_buffer.clear()
        self._silence_counter = 0

        if not self.stt or not self._on_query:
            self._set_state(PipelineState.LISTENING)
            return

        text, lang = self.stt.transcribe_bytes(audio_bytes, self._sample_rate)
        if not text.strip():
            self._set_state(PipelineState.LISTENING)
            return

        logger.info("STT: '%s' (lang=%s)", text, lang)
        self._set_state(PipelineState.SPEAKING)
        self._should_play.set()

        try:
            sentence_buffer = []
            for token in self._on_query(text):
                sentence_buffer.append(token)
                joined = "".join(sentence_buffer)
                if any(joined.rstrip().endswith(p) for p in ".!?"):
                    self._synthesize_and_queue(joined.strip(), lang)
                    sentence_buffer.clear()
            remaining = "".join(sentence_buffer).strip()
            if remaining:
                self._synthesize_and_queue(remaining, lang)
        except Exception:
            logger.exception("Brain query failed")

        if self._on_response_complete:
            self._on_response_complete()
        if self._mode in ("conversational", "hybrid"):
            self._set_state(PipelineState.LISTENING)
        else:
            self._set_state(PipelineState.IDLE)

    def _synthesize_and_queue(self, text: str, lang: str) -> None:
        if not self.tts_selector or not text:
            return
        try:
            engine = self.tts_selector.select(text, detected_lang=lang)
            for chunk in engine.synthesize_stream(text):
                if self._stop_event.is_set() or not self._should_play.is_set():
                    break
                self._playback_queue.put(chunk)
        except Exception:
            logger.exception("TTS failed for: %s", text[:50])

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()


def is_available() -> bool:
    """Return True if the minimum deps for the voice pipeline exist."""
    return True


__all__ = ["VoicePipeline", "PipelineState", "is_available"]
