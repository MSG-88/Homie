from __future__ import annotations
import logging
from typing import Iterator, Optional
from homie_core.voice.tts import BaseTTS

logger = logging.getLogger(__name__)

# Optional dependency check
try:
    from kokoro import KPipeline as _KPipeline
    _HAS_KOKORO = True
except ImportError:
    _KPipeline = None  # type: ignore[misc,assignment]
    _HAS_KOKORO = False

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


class KokoroTTS(BaseTTS):
    """High-quality neural TTS using the Kokoro library.

    Kokoro produces natural-sounding speech with support for multiple languages
    and voices.  When the ``kokoro`` package is not installed the class
    instantiates successfully but ``load()`` will be a no-op and synthesis
    methods return empty results.

    Parameters
    ----------
    voice : str
        Voice identifier (e.g. "af_heart", "af_bella", "am_adam").
    lang : str
        Language code ("en", "fr", "es", "de", "it", "pt", "ja", "zh").
    speed : float
        Speech speed multiplier.  1.0 = normal, <1.0 = slower, >1.0 = faster.
    sample_rate : int
        Output sample rate in Hz (Kokoro native rate is 24000).
    """

    _LANG_MAP = {
        "en": "a", "gb": "b", "fr": "f", "es": "e",
        "de": "d", "it": "i", "pt": "p", "ja": "j", "zh": "z",
    }

    def __init__(
        self,
        voice: str = "af_heart",
        lang: str = "en",
        speed: float = 1.0,
        sample_rate: int = 24000,
    ) -> None:
        self._voice = voice
        self._lang = lang
        self._speed = max(0.1, min(speed, 3.0))  # clamp to safe range
        self._sample_rate = sample_rate
        self._pipeline = None

    def load(self, device: str = "cuda") -> None:
        """Load the Kokoro pipeline.

        Parameters
        ----------
        device : str
            "cuda" or "cpu".  Falls back to "cpu" automatically if CUDA is
            unavailable.
        """
        if not _HAS_KOKORO:
            logger.warning(
                "kokoro not installed — KokoroTTS disabled.  "
                "Install with: pip install kokoro"
            )
            return
        try:
            lang_code = self._LANG_MAP.get(self._lang, "a")
            self._pipeline = _KPipeline(lang_code=lang_code, device=device)
            logger.info(
                "KokoroTTS loaded: voice=%s lang=%s speed=%.1f device=%s",
                self._voice, self._lang, self._speed, device,
            )
        except Exception:
            logger.exception("Failed to load KokoroTTS")
            self._pipeline = None

    def synthesize(self, text: str) -> bytes:
        """Synthesize speech from text, returning raw PCM16 bytes.

        Returns empty bytes if the pipeline is not loaded or synthesis fails.
        """
        if not self._pipeline:
            return b""
        if not _HAS_NUMPY:
            logger.warning("numpy not installed — cannot synthesize")
            return b""
        if not text.strip():
            return b""
        try:
            chunks = []
            for _, _, audio in self._pipeline(
                text, voice=self._voice, speed=self._speed
            ):
                chunks.append(
                    (audio * 32767).astype(_np.int16).tobytes()
                )
            return b"".join(chunks)
        except Exception:
            logger.exception("KokoroTTS synthesis failed for: %s", text[:80])
            return b""

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Stream synthesized audio chunks for lower latency playback.

        Yields raw PCM16 byte chunks as they become available.
        """
        if not self._pipeline:
            return
        if not _HAS_NUMPY:
            logger.warning("numpy not installed — cannot synthesize")
            return
        if not text.strip():
            return
        try:
            for _, _, audio in self._pipeline(
                text, voice=self._voice, speed=self._speed
            ):
                yield (audio * 32767).astype(_np.int16).tobytes()
        except Exception:
            logger.exception("KokoroTTS streaming failed for: %s", text[:80])

    def set_voice(self, voice: str) -> None:
        """Change the active voice without reloading the pipeline."""
        self._voice = voice

    def set_speed(self, speed: float) -> None:
        """Change speech speed (clamped to 0.1-3.0)."""
        self._speed = max(0.1, min(speed, 3.0))

    def unload(self) -> None:
        """Release the pipeline from memory."""
        self._pipeline = None
        logger.debug("KokoroTTS unloaded")

    @property
    def supported_languages(self) -> list[str]:
        return list(self._LANG_MAP.keys())

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def is_available(self) -> bool:
        """True if the kokoro library is installed."""
        return _HAS_KOKORO

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def speed(self) -> float:
        return self._speed

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"KokoroTTS(voice={self._voice!r}, lang={self._lang!r}, "
            f"speed={self._speed:.1f}, {status})"
        )
