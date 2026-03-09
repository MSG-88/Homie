from __future__ import annotations

from typing import Optional


class SpeechToText:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        except ImportError:
            pass

    def transcribe(self, audio_path: str) -> str:
        if not self._model:
            return ""
        segments, _ = self._model.transcribe(audio_path, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if not self._model:
            return ""
        try:
            import numpy as np
            import io
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(audio, beam_size=5)
            return " ".join(seg.text for seg in segments).strip()
        except Exception:
            return ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
