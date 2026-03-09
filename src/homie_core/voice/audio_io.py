from __future__ import annotations

import threading
from typing import Callable, Optional


class AudioIO:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._recording = False
        self._playing = False
        self._stream = None
        self._lock = threading.Lock()

    def start_recording(self, callback: Callable[[bytes], None]) -> None:
        if self._recording:
            return
        self._recording = True
        self._record_thread = threading.Thread(target=self._record_loop, args=(callback,), daemon=True)
        self._record_thread.start()

    def stop_recording(self) -> None:
        self._recording = False

    def play_audio(self, audio_data: bytes, sample_rate: int | None = None) -> None:
        with self._lock:
            self._playing = True
        try:
            import sounddevice as sd
            import numpy as np
            sr = sample_rate or self.sample_rate
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_array, sr)
            sd.wait()
        except ImportError:
            pass
        finally:
            with self._lock:
                self._playing = False

    def stop_playback(self) -> None:
        with self._lock:
            self._playing = False
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _record_loop(self, callback: Callable[[bytes], None]) -> None:
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=self.channels,
                           rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
            while self._recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                callback(data)
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except ImportError:
            self._recording = False
        except Exception:
            self._recording = False
