from __future__ import annotations
import logging
from typing import Optional
from homie_core.voice.tts import BaseTTS

logger = logging.getLogger(__name__)


class TTSSelector:
    WORD_THRESHOLD = 20

    def __init__(self, fast: Optional[BaseTTS] = None, quality: Optional[BaseTTS] = None,
                 multilingual: Optional[BaseTTS] = None, mode: str = "auto") -> None:
        self._fast = fast
        self._quality = quality
        self._multilingual = multilingual
        self._mode = mode
        self._engines = {"fast": fast, "quality": quality, "multilingual": multilingual}

    def select(self, text: str, detected_lang: str = "en") -> BaseTTS:
        if self._mode != "auto":
            engine = self._engines.get(self._mode)
            if engine and engine.is_loaded:
                return engine

        if detected_lang != "en" and self._multilingual and self._multilingual.is_loaded:
            return self._multilingual
        if len(text.split()) < self.WORD_THRESHOLD and self._fast and self._fast.is_loaded:
            return self._fast
        if self._quality and self._quality.is_loaded:
            return self._quality
        if self._fast and self._fast.is_loaded:
            return self._fast
        if self._multilingual and self._multilingual.is_loaded:
            return self._multilingual
        raise RuntimeError("No TTS engine available")

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    @property
    def available_engines(self) -> dict[str, bool]:
        return {n: (e is not None and e.is_loaded) for n, e in self._engines.items()}
