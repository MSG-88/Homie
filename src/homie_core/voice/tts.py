from __future__ import annotations

from pathlib import Path
from typing import Optional


class TextToSpeech:
    def __init__(self, voice: str = "default"):
        self.voice = voice
        self._model = None

    def load(self) -> None:
        try:
            # Piper TTS loading
            pass  # Loaded on demand
        except ImportError:
            pass

    def synthesize(self, text: str) -> bytes:
        try:
            import piper
            if not self._model:
                self._model = piper.PiperVoice.load(self.voice)
            audio = b""
            for chunk in self._model.synthesize_stream_raw(text):
                audio += chunk
            return audio
        except (ImportError, Exception):
            return b""

    def synthesize_to_file(self, text: str, path: str | Path) -> bool:
        audio = self.synthesize(text)
        if audio:
            Path(path).write_bytes(audio)
            return True
        return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
