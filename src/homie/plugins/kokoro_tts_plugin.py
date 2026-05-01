"""Kokoro-TTS voice output plugin for Homie.

Provides lightweight, offline text-to-speech synthesis using the Kokoro TTS
model (hexgrad/Kokoro-82M). Kokoro is a small (~82M param) model that produces
high-quality speech locally without network calls. It supports multiple voices
and speaking styles.

Usage:
    from homie.plugins.kokoro_tts_plugin import KokoroTTSPlugin

    plugin = KokoroTTSPlugin()
    plugin.activate()
    plugin.speak("Hello from Homie!")
    plugin.deactivate()
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "hexgrad/Kokoro-82M"
DEFAULT_VOICE = "af_heart"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SPEED = 1.0


@dataclass
class KokoroTTSConfig:
    """Configuration for the Kokoro TTS plugin."""

    model_repo: str = DEFAULT_MODEL_REPO
    voice: str = DEFAULT_VOICE
    sample_rate: int = DEFAULT_SAMPLE_RATE
    speed: float = DEFAULT_SPEED
    output_dir: Optional[str] = None
    auto_play: bool = True
    lang_code: str = "a"
    device: str = "cpu"


class KokoroTTSPlugin:
    """Offline text-to-speech plugin using Kokoro TTS for Homie.

    Kokoro is a lightweight (~82M parameter) TTS model that runs locally
    on CPU or GPU, producing natural-sounding speech without any network
    dependency.
    """

    name: str = "kokoro_tts"
    version: str = "1.0.0"

    def __init__(self, config: Optional[KokoroTTSConfig] = None) -> None:
        self._config = config or KokoroTTSConfig()
        self._pipeline: Any = None
        self._active = False
        self._output_dir = (
            Path(self._config.output_dir)
            if self._config.output_dir
            else Path(tempfile.gettempdir()) / "homie_tts"
        )

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Load the Kokoro TTS model and prepare for synthesis."""
        if self._active:
            logger.info("KokoroTTSPlugin already active")
            return

        try:
            from kokoro import KPipeline  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro TTS is not installed. "
                "Install with: pip install kokoro soundfile"
            ) from exc

        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading Kokoro TTS pipeline (voice=%s, device=%s)",
            self._config.voice,
            self._config.device,
        )
        self._pipeline = KPipeline(
            lang_code=self._config.lang_code,
            repo_id=self._config.model_repo,
            device=self._config.device,
        )
        self._active = True
        logger.info("KokoroTTSPlugin activated")

    def deactivate(self) -> None:
        """Release the model and free resources."""
        self._pipeline = None
        self._active = False
        logger.info("KokoroTTSPlugin deactivated")

    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Path:
        """Synthesize speech from text and save to a WAV file.

        Args:
            text: The text to convert to speech.
            output_path: Where to save the WAV file. If ``None``, a
                temporary file is created in the plugin output directory.

        Returns:
            Path to the generated WAV file.

        Raises:
            RuntimeError: If the plugin has not been activated.
            ValueError: If *text* is empty.
        """
        if not self._active or self._pipeline is None:
            raise RuntimeError("Plugin is not active. Call activate() first.")
        if not text or not text.strip():
            raise ValueError("Text must be non-empty.")

        import soundfile as sf  # type: ignore[import-untyped]

        if output_path is None:
            output_path = self._output_dir / "homie_tts_output.wav"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug("Synthesizing %d chars with voice=%s", len(text), self._config.voice)

        audio_chunks: List[Any] = []
        for _graphemes, _phonemes, audio_chunk in self._pipeline(
            text,
            voice=self._config.voice,
            speed=self._config.speed,
        ):
            if audio_chunk is not None:
                audio_chunks.append(audio_chunk)

        if not audio_chunks:
            raise RuntimeError("Kokoro TTS produced no audio output.")

        # Concatenate chunks
        try:
            import numpy as np  # type: ignore[import-untyped]
            combined = np.concatenate(audio_chunks)
        except ImportError:
            # Fallback: use the last chunk only
            combined = audio_chunks[-1]

        sf.write(str(output_path), combined, self._config.sample_rate)
        logger.info("Saved TTS audio to %s", output_path)
        return output_path

    def speak(self, text: str) -> Path:
        """Synthesize and optionally play the audio.

        This is the primary method for Homie voice output. It synthesizes
        the text and, if ``auto_play`` is enabled, attempts to play the
        audio through the system's default audio player.

        Args:
            text: The text to speak.

        Returns:
            Path to the generated WAV file.
        """
        wav_path = self.synthesize(text)
        if self._config.auto_play:
            _play_audio(wav_path)
        return wav_path

    def list_voices(self) -> List[str]:
        """Return a list of known Kokoro voice identifiers."""
        return [
            "af_heart",
            "af_alloy",
            "af_aoede",
            "af_bella",
            "af_jessica",
            "af_kore",
            "af_nicole",
            "af_nova",
            "af_river",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_echo",
            "am_eric",
            "am_fenrir",
            "am_liam",
            "am_michael",
            "am_onyx",
            "am_puck",
            "am_santa",
        ]

    def set_voice(self, voice: str) -> None:
        """Change the active voice.

        Args:
            voice: A valid Kokoro voice identifier.
        """
        self._config.voice = voice
        logger.info("Voice changed to %s", voice)

    def get_config(self) -> Dict[str, Any]:
        """Return the current plugin configuration as a dict."""
        return {
            "model_repo": self._config.model_repo,
            "voice": self._config.voice,
            "sample_rate": self._config.sample_rate,
            "speed": self._config.speed,
            "output_dir": str(self._output_dir),
            "auto_play": self._config.auto_play,
            "lang_code": self._config.lang_code,
            "device": self._config.device,
        }


def _play_audio(path: Path) -> None:
    """Best-effort local audio playback."""
    import platform

    system = platform.system().lower()
    try:
        if system == "windows":
            import winsound  # type: ignore[import-untyped]
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
        elif system == "darwin":
            subprocess.run(["afplay", str(path)], check=False)
        else:
            # Linux: try aplay, then paplay, then ffplay
            for player in ("aplay", "paplay", "ffplay"):
                try:
                    cmd = [player, str(path)]
                    if player == "ffplay":
                        cmd = [player, "-nodisp", "-autoexit", str(path)]
                    subprocess.run(cmd, check=True, capture_output=True)
                    return
                except FileNotFoundError:
                    continue
            logger.warning("No audio player found. Install aplay, paplay, or ffplay.")
    except Exception:  # noqa: BLE001
        logger.warning("Audio playback failed for %s", path)


def register() -> Dict[str, Any]:
    """Register this plugin with the Homie plugin system."""
    return {
        "name": KokoroTTSPlugin.name,
        "version": KokoroTTSPlugin.version,
        "class": KokoroTTSPlugin,
        "description": (
            "Offline text-to-speech using Kokoro TTS (~82M params). "
            "Produces natural speech locally on CPU or GPU."
        ),
        "dependencies": ["kokoro", "soundfile"],
    }
