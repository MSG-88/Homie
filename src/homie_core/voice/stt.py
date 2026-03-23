from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Optional dependency check
try:
    from faster_whisper import WhisperModel as _WhisperModel
    _HAS_FASTER_WHISPER = True
except ImportError:
    _WhisperModel = None  # type: ignore[misc,assignment]
    _HAS_FASTER_WHISPER = False

# Optional numpy check
try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# Supported model sizes with descriptions for validation
SUPPORTED_MODELS = {
    "tiny": "Fastest, lowest accuracy (~39M params)",
    "tiny.en": "Fastest, English-only (~39M params)",
    "base": "Fast, decent accuracy (~74M params)",
    "base.en": "Fast, English-only (~74M params)",
    "small": "Balanced speed/accuracy (~244M params)",
    "small.en": "Balanced, English-only (~244M params)",
    "medium": "High accuracy, slower (~769M params)",
    "medium.en": "High accuracy, English-only (~769M params)",
    "large-v2": "Best accuracy, slowest (~1550M params)",
    "large-v3": "Latest best accuracy (~1550M params)",
}


class SpeechToText:
    """Speech-to-text engine using faster-whisper.

    All heavy dependencies (faster_whisper, numpy) are optional.  When they are
    not installed, the class still instantiates successfully but ``load()``
    will log a warning and ``transcribe*`` methods return empty results.

    Parameters
    ----------
    model_size : str
        One of the faster-whisper model identifiers (e.g. "tiny.en", "base.en",
        "small", "medium", "large-v3").
    device : str
        "cuda" or "cpu".
    compute_type : str
        Quantisation type passed to faster-whisper ("float16", "float32",
        "int8", "int8_float16", etc.).
    beam_size : int
        Beam size for decoding.  Smaller = faster, larger = more accurate.
    language : str or None
        If set, forces the language instead of auto-detecting.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "float32",
        beam_size: int = 5,
        language: Optional[str] = None,
    ) -> None:
        if model_size not in SUPPORTED_MODELS:
            logger.warning(
                "Unknown model_size '%s', supported: %s",
                model_size,
                ", ".join(SUPPORTED_MODELS),
            )
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model = None

    def load(self) -> None:
        """Load the whisper model.  No-op if faster-whisper is not installed."""
        if not _HAS_FASTER_WHISPER:
            logger.warning(
                "faster-whisper not installed — STT disabled.  "
                "Install with: pip install faster-whisper"
            )
            return
        try:
            self._model = _WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(
                "STT loaded: model=%s device=%s compute=%s",
                self.model_size,
                self.device,
                self.compute_type,
            )
        except Exception:
            logger.exception("Failed to load STT model '%s'", self.model_size)
            self._model = None

    def transcribe(self, audio_path: str) -> tuple[str, str]:
        """Transcribe audio from a file path.

        Returns
        -------
        tuple[str, str]
            (transcribed_text, detected_language).  Returns ("", "en") on
            failure or if model is not loaded.
        """
        if not self._model:
            return "", "en"
        try:
            segments, info = self._model.transcribe(
                audio_path,
                beam_size=self.beam_size,
                language=self.language,
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text, info.language
        except Exception:
            logger.exception("STT transcription failed for file: %s", audio_path)
            return "", "en"

    def transcribe_bytes(
        self, audio_bytes: bytes, sample_rate: int = 16000
    ) -> tuple[str, str]:
        """Transcribe raw PCM audio bytes (16-bit signed LE).

        Parameters
        ----------
        audio_bytes : bytes
            Raw PCM data — 16-bit signed little-endian samples.
        sample_rate : int
            Sample rate of the audio (default 16000 Hz).

        Returns
        -------
        tuple[str, str]
            (transcribed_text, detected_language).  Returns ("", "en") on
            failure or if model/numpy is not available.
        """
        if not self._model:
            return "", "en"
        if not _HAS_NUMPY:
            logger.warning("numpy not installed — cannot transcribe bytes")
            return "", "en"
        try:
            audio_np = (
                _np.frombuffer(audio_bytes, dtype=_np.int16)
                .astype(_np.float32)
                / 32768.0
            )
            if len(audio_np) == 0:
                return "", "en"
            segments, info = self._model.transcribe(
                audio_np,
                beam_size=self.beam_size,
                language=self.language,
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text, info.language
        except Exception:
            logger.exception("STT byte transcription failed")
            return "", "en"

    @property
    def is_loaded(self) -> bool:
        """True if the whisper model has been successfully loaded."""
        return self._model is not None

    @property
    def is_available(self) -> bool:
        """True if the faster-whisper library is installed."""
        return _HAS_FASTER_WHISPER

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        logger.debug("STT model unloaded")

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"SpeechToText(model={self.model_size!r}, device={self.device!r}, {status})"
