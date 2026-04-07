"""Homie plugin for Kokoro-TTS local text-to-speech.

Integrates the Kokoro-TTS model (hexgrad/Kokoro-82M) for high-quality,
fully offline voice output. Kokoro is a lightweight 82M-parameter TTS
model that runs efficiently on CPU, making it ideal for local-first
assistants like Homie.

Requires:
    pip install kokoro soundfile

Model files are downloaded once on first use and cached locally.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_OUTPUT_DIR = Path.home() / ".homie" / "tts_cache"


@dataclass
class KokoroTTSConfig:
    """Configuration for the Kokoro-TTS plugin."""

    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED
    lang_code: str = "a"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    sample_rate: int = DEFAULT_SAMPLE_RATE
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KokoroTTSConfig:
        """Build config from a plain dict (e.g. from homie.config.yaml)."""
        return cls(
            voice=data.get("voice", DEFAULT_VOICE),
            speed=float(data.get("speed", DEFAULT_SPEED)),
            lang_code=data.get("lang_code", "a"),
            output_dir=Path(data.get("output_dir", DEFAULT_OUTPUT_DIR)),
            sample_rate=int(data.get("sample_rate", DEFAULT_SAMPLE_RATE)),
            enabled=bool(data.get("enabled", True)),
        )


class KokoroTTSPlugin:
    """Local text-to-speech plugin using Kokoro-TTS (hexgrad/Kokoro-82M).

    The model is loaded lazily on first synthesis request and kept in
    memory for subsequent calls.  All inference is local â€” no network
    calls are made after the one-time model download.
    """

    name: str = "kokoro_tts"
    version: str = "0.1.0"

    def __init__(self, config: Optional[KokoroTTSConfig] = None) -> None:
        self._config = config or KokoroTTSConfig()
        self._pipeline: Any = None  # lazy-loaded KPipeline instance
        self._lock = threading.Lock()
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Activate the plugin and ensure the output directory exists."""
        if not self._config.enabled:
            logger.info("KokoroTTS plugin is disabled via config.")
            return
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        self._active = True
        logger.info(
            "KokoroTTS plugin activated (voice=%s, speed=%.1f)",
            self._config.voice,
            self._config.speed,
        )

    def deactivate(self) -> None:
        """Release model resources."""
        self._pipeline = None
        self._active = False
        logger.info("KokoroTTS plugin deactivated.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_pipeline(self) -> Any:
        """Lazy-load the Kokoro pipeline (thread-safe)."""
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                from kokoro import KPipeline  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "kokoro package not installed. "
                    "Run: pip install kokoro soundfile"
                ) from exc
            logger.info(
                "Loading Kokoro pipeline (lang_code=%s) â€¦",
                self._config.lang_code,
            )
            self._pipeline = KPipeline(lang_code=self._config.lang_code)
            logger.info("Kokoro pipeline loaded.")
            return self._pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Synthesize *text* to a WAV file and return its path.

        Args:
            text: The input text to speak.
            voice: Override the default voice for this call.
            speed: Override the default speed for this call.
            save_path: Explicit output file path.  When *None* a
                deterministic filename is generated inside
                ``output_dir``.

        Returns:
            Path to the generated ``.wav`` file.
        """
        if not self._active:
            raise RuntimeError("Plugin is not active. Call activate() first.")

        try:
            import soundfile as sf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "soundfile package not installed. "
                "Run: pip install soundfile"
            ) from exc

        pipeline = self._ensure_pipeline()
        effective_voice = voice or self._config.voice
        effective_speed = speed if speed is not None else self._config.speed

        if save_path is None:
            import hashlib

            tag = hashlib.sha256(
                f"{text}|{effective_voice}|{effective_speed}".encode()
            ).hexdigest()[:16]
            save_path = self._config.output_dir / f"kokoro_{tag}.wav"

        # Kokoro yields (graphemes, phonemes, audio) generator entries.
        # We concatenate all audio chunks into a single file.
        import numpy as np  # type: ignore[import-untyped]

        chunks: List[Any] = []
        for _gs, _ps, audio in pipeline(
            text, voice=effective_voice, speed=effective_speed
        ):
            if audio is not None:
                chunks.append(audio)

        if not chunks:
            raise RuntimeError("Kokoro produced no audio output.")

        combined = np.concatenate(chunks)
        sf.write(str(save_path), combined, self._config.sample_rate)
        logger.info("TTS audio saved to %s", save_path)
        return save_path

    def speak(self, text: str, **kwargs: Any) -> Path:
        """Convenience alias for :meth:`synthesize`."""
        return self.synthesize(text, **kwargs)

    def list_voices(self) -> List[str]:
        """Return a curated list of known Kokoro voice identifiers."""
        return [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ]


# ------------------------------------------------------------------
# Module-level convenience for Homie plugin discovery
# ------------------------------------------------------------------

_instance: Optional[KokoroTTSPlugin] = None


def register(config_dict: Optional[Dict[str, Any]] = None) -> KokoroTTSPlugin:
    """Create, activate, and return the singleton plugin instance.

    Called by the Homie plugin loader at startup.  Pass the
    ``tts.kokoro`` section of ``homie.config.yaml`` as *config_dict*.
    """
    global _instance
    cfg = KokoroTTSConfig.from_dict(config_dict or {})
    _instance = KokoroTTSPlugin(config=cfg)
    _instance.activate()
    return _instance


__all__ = [
    "KokoroTTSConfig",
    "KokoroTTSPlugin",
    "register",
]
