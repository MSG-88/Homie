from __future__ import annotations
import logging
from typing import Iterator, Optional
from homie_core.voice.tts import BaseTTS

logger = logging.getLogger(__name__)


class MeloTTS(BaseTTS):
    def __init__(self, language: str = "EN", speaker_id: int = 0) -> None:
        self._language = language.upper()
        self._speaker_id = speaker_id
        self._model = None

    def load(self, device: str = "cuda") -> None:
        try:
            from melo.api import TTS as MeloAPI
            self._model = MeloAPI(language=self._language, device=device)
        except ImportError:
            logger.warning("melo-tts not installed")
        except Exception:
            logger.exception("Failed to load MeloTTS")

    def synthesize(self, text: str) -> bytes:
        if not self._model:
            return b""
        try:
            import numpy as np
            import tempfile
            import os
            import wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                self._model.tts_to_file(text, self._speaker_id, tmp_path, quiet=True)
                with wave.open(tmp_path, "rb") as wf:
                    return wf.readframes(wf.getnframes())
            finally:
                os.unlink(tmp_path)
        except Exception:
            logger.exception("MeloTTS synthesis failed")
            return b""

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        data = self.synthesize(text)
        if data:
            yield data

    def unload(self) -> None:
        self._model = None

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "fr", "es", "de", "it", "pt", "zh", "ja", "ko",
                "hi", "ta", "te", "ml", "bn", "gu", "kn", "mr", "pa"]

    @property
    def name(self) -> str:
        return "melo"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
