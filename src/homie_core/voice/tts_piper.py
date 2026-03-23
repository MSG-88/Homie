from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, Optional
from homie_core.voice.tts import BaseTTS

logger = logging.getLogger(__name__)

# Optional dependency check
try:
    import piper as _piper
    _HAS_PIPER = True
except ImportError:
    _piper = None  # type: ignore[assignment]
    _HAS_PIPER = False


class PiperTTS(BaseTTS):
    """Lightweight, fast TTS using the Piper library.

    Piper is a CPU-friendly neural TTS that runs well even on low-end hardware.
    It serves as the primary fast/fallback engine when heavier models like
    Kokoro are unavailable.

    When the ``piper`` package is not installed, the class instantiates
    successfully but ``load()`` will be a no-op and synthesis methods return
    empty results.

    Parameters
    ----------
    voice : str
        Path to a Piper ONNX voice model, or a voice name that Piper can
        resolve (e.g. "en_US-lessac-medium").
    speaker_id : int or None
        Speaker ID for multi-speaker models.  ``None`` uses the default.
    length_scale : float
        Controls speech speed.  1.0 = normal, <1.0 = faster, >1.0 = slower.
    sample_rate : int
        Expected output sample rate (Piper models are typically 22050 Hz).
    """

    def __init__(
        self,
        voice: str = "default",
        speaker_id: Optional[int] = None,
        length_scale: float = 1.0,
        sample_rate: int = 22050,
    ) -> None:
        self._voice_path = voice
        self._speaker_id = speaker_id
        self._length_scale = max(0.1, min(length_scale, 3.0))
        self._sample_rate = sample_rate
        self._model = None

    def load(self, device: str = "cpu") -> None:
        """Load the Piper voice model.

        Parameters
        ----------
        device : str
            Ignored for Piper (always CPU), accepted for interface
            compatibility.
        """
        if not _HAS_PIPER:
            logger.warning(
                "piper-tts not installed — PiperTTS disabled.  "
                "Install with: pip install piper-tts"
            )
            return
        try:
            self._model = _piper.PiperVoice.load(self._voice_path)
            logger.info("PiperTTS loaded: voice=%s", self._voice_path)
        except FileNotFoundError:
            logger.error(
                "Piper voice model not found: %s", self._voice_path
            )
            self._model = None
        except Exception:
            logger.exception("Failed to load PiperTTS")
            self._model = None

    def synthesize(self, text: str) -> bytes:
        """Synthesize speech from text, returning raw PCM16 bytes.

        Returns empty bytes if the model is not loaded or synthesis fails.
        """
        if not self._model:
            return b""
        if not text.strip():
            return b""
        try:
            audio = b""
            synth_kwargs = {}
            if self._speaker_id is not None:
                synth_kwargs["speaker_id"] = self._speaker_id
            if self._length_scale != 1.0:
                synth_kwargs["length_scale"] = self._length_scale
            for chunk in self._model.synthesize_stream_raw(
                text, **synth_kwargs
            ):
                audio += chunk
            return audio
        except Exception:
            logger.exception("PiperTTS synthesis failed for: %s", text[:80])
            return b""

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Stream synthesized audio chunks for lower latency playback.

        Yields raw PCM16 byte chunks as they become available.
        """
        if not self._model:
            return
        if not text.strip():
            return
        try:
            synth_kwargs = {}
            if self._speaker_id is not None:
                synth_kwargs["speaker_id"] = self._speaker_id
            if self._length_scale != 1.0:
                synth_kwargs["length_scale"] = self._length_scale
            for chunk in self._model.synthesize_stream_raw(
                text, **synth_kwargs
            ):
                yield chunk
        except Exception:
            logger.exception("PiperTTS streaming failed for: %s", text[:80])

    def synthesize_to_file(self, text: str, path: str | Path) -> bool:
        """Synthesize to a WAV file.  Returns True on success."""
        data = self.synthesize(text)
        if data:
            Path(path).write_bytes(data)
            return True
        return False

    def set_length_scale(self, scale: float) -> None:
        """Change speech speed (clamped to 0.1-3.0)."""
        self._length_scale = max(0.1, min(scale, 3.0))

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        logger.debug("PiperTTS unloaded")

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "fr", "es", "de", "it", "pt"]

    @property
    def name(self) -> str:
        return "piper"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_available(self) -> bool:
        """True if the piper-tts library is installed."""
        return _HAS_PIPER

    @property
    def voice(self) -> str:
        return self._voice_path

    @property
    def length_scale(self) -> float:
        return self._length_scale

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"PiperTTS(voice={self._voice_path!r}, "
            f"length_scale={self._length_scale:.1f}, {status})"
        )
