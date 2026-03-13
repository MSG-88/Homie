from __future__ import annotations
import logging
from typing import Iterator, Optional
from homie_core.voice.tts import BaseTTS

logger = logging.getLogger(__name__)


class KokoroTTS(BaseTTS):
    _LANG_MAP = {
        "en": "a", "gb": "b", "fr": "f", "es": "e",
        "de": "d", "it": "i", "pt": "p", "ja": "j", "zh": "z",
    }

    def __init__(self, voice: str = "af_heart", lang: str = "en") -> None:
        self._voice = voice
        self._lang = lang
        self._pipeline = None

    def load(self, device: str = "cuda") -> None:
        try:
            from kokoro import KPipeline
            lang_code = self._LANG_MAP.get(self._lang, "a")
            self._pipeline = KPipeline(lang_code=lang_code, device=device)
        except ImportError:
            logger.warning("kokoro not installed")
        except Exception:
            logger.exception("Failed to load KokoroTTS")

    def synthesize(self, text: str) -> bytes:
        if not self._pipeline:
            return b""
        try:
            import numpy as np
            chunks = []
            for _, _, audio in self._pipeline(text, voice=self._voice):
                chunks.append((audio * 32767).astype(np.int16).tobytes())
            return b"".join(chunks)
        except Exception:
            logger.exception("KokoroTTS synthesis failed")
            return b""

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        if not self._pipeline:
            return
        try:
            import numpy as np
            for _, _, audio in self._pipeline(text, voice=self._voice):
                yield (audio * 32767).astype(np.int16).tobytes()
        except Exception:
            logger.exception("KokoroTTS streaming failed")

    def unload(self) -> None:
        self._pipeline = None

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "fr", "es", "de", "it", "pt", "ja", "zh"]

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
