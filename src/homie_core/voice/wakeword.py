from __future__ import annotations

import threading
from typing import Callable, Optional


class WakeWordEngine:
    def __init__(self, wake_word: str = "hey homie"):
        self.wake_word = wake_word
        self._running = False
        self._callback: Optional[Callable[[], None]] = None
        self._model = None

    def start(self, on_wake: Callable[[], None]) -> None:
        self._callback = on_wake
        self._running = True
        try:
            from openwakeword.model import Model
            self._model = Model(inference_framework="onnx")
        except ImportError:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self._model = None

    def process_audio(self, audio_chunk: bytes) -> bool:
        if not self._running or not self._model:
            return False
        try:
            import numpy as np
            audio = np.frombuffer(audio_chunk, dtype=np.int16)
            predictions = self._model.predict(audio)
            for key, score in predictions.items():
                if score > 0.5:
                    if self._callback:
                        self._callback()
                    return True
        except Exception:
            pass
        return False

    @property
    def is_running(self) -> bool:
        return self._running
