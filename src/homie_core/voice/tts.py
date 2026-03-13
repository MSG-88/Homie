from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    @abstractmethod
    def load(self, device: str = "cpu") -> None: ...

    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[bytes]: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def is_loaded(self) -> bool:
        return False


class PiperTTS(BaseTTS):
    def __init__(self, voice: str = "default") -> None:
        self._voice = voice
        self._model = None

    def load(self, device: str = "cpu") -> None:
        try:
            import piper
            self._model = piper.PiperVoice.load(self._voice)
        except ImportError:
            logger.error("piper-tts not installed")
        except Exception:
            logger.exception("Failed to load PiperTTS")

    def synthesize(self, text: str) -> bytes:
        if not self._model:
            return b""
        try:
            audio = b""
            for chunk in self._model.synthesize_stream_raw(text):
                audio += chunk
            return audio
        except Exception:
            logger.exception("PiperTTS synthesis failed")
            return b""

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        if not self._model:
            return
        try:
            for chunk in self._model.synthesize_stream_raw(text):
                yield chunk
        except Exception:
            logger.exception("PiperTTS streaming failed")

    def unload(self) -> None:
        self._model = None

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "fr", "es", "de", "it", "pt"]

    @property
    def name(self) -> str:
        return "piper"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def synthesize_to_file(self, text: str, path: str | Path) -> bool:
        data = self.synthesize(text)
        if data:
            Path(path).write_bytes(data)
            return True
        return False


# ---------------------------------------------------------------------------
# Legacy alias kept for backward compatibility with existing code
# ---------------------------------------------------------------------------

class TextToSpeech(PiperTTS):
    """Backward-compatible alias for PiperTTS.

    Existing code that instantiates ``TextToSpeech(voice=...)`` continues to
    work.  The ``voice`` kwarg is forwarded to ``PiperTTS.__init__``.
    """

    def __init__(self, voice: str = "default") -> None:
        super().__init__(voice=voice)
        # Expose ``voice`` attribute for callers that read it directly.
        self.voice = voice
