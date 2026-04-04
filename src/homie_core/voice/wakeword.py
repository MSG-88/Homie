from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


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
        except Exception as e:
            logger.warning("Wake-word audio processing failed: %s", e)
        return False

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Lightweight text-based wake-word detector (used by VoicePipeline)
# ---------------------------------------------------------------------------

DEFAULT_WAKE_PHRASE = "hey homie"


class WakeWordDetector:
    """Detects the wake phrase in transcribed text snippets.

    Parameters
    ----------
    wake_phrase : str
        The phrase to listen for (case-insensitive).
    on_detected : callable, optional
        Callback invoked (no args) when wake word is detected.
    """

    def __init__(
        self,
        wake_phrase: str = DEFAULT_WAKE_PHRASE,
        on_detected: Optional[Callable[[], None]] = None,
    ) -> None:
        self.wake_phrase = wake_phrase.lower().strip()
        self.on_detected = on_detected

    def check(self, text: str) -> bool:
        """Return True if *text* contains the wake phrase."""
        if self.wake_phrase in text.lower():
            if self.on_detected is not None:
                self.on_detected()
            return True
        return False
