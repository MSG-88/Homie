from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(self, energy_threshold: float = 500.0, silence_frames: int = 30):
        self.energy_threshold = energy_threshold
        self.silence_frames = silence_frames
        self._silent_count: int = 0
        self._is_speaking: bool = False
        self._energy_history: deque = deque(maxlen=100)

    def process(self, audio_chunk: bytes) -> bool:
        try:
            import numpy as np
            audio = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = float(np.sqrt(np.mean(audio.astype(float) ** 2)))
        except (ImportError, ValueError):
            energy = 0.0

        self._energy_history.append(energy)

        if energy > self.energy_threshold:
            self._is_speaking = True
            self._silent_count = 0
        else:
            self._silent_count += 1
            if self._silent_count >= self.silence_frames:
                self._is_speaking = False

        return self._is_speaking

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        self._silent_count = 0
        self._is_speaking = False
        self._energy_history.clear()

    def get_average_energy(self) -> float:
        if not self._energy_history:
            return 0.0
        return sum(self._energy_history) / len(self._energy_history)


# ---------------------------------------------------------------------------
# Lightweight VAD wrapper (used by VoicePipeline)
# ---------------------------------------------------------------------------

import struct as _struct
from typing import Optional as _Optional

try:
    import webrtcvad as _webrtcvad  # type: ignore[import-untyped]
    _HAS_WEBRTCVAD = True
except ImportError:
    _HAS_WEBRTCVAD = False


class VAD:
    """Wraps webrtcvad or falls back to energy-based detection.

    Parameters
    ----------
    aggressiveness : int
        0-3, where 3 is the most aggressive at filtering non-speech.
    sample_rate : int
        Must be 8000, 16000, 32000, or 48000 for webrtcvad.
    energy_threshold : int
        RMS threshold for the energy-based fallback.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = 16_000,
        energy_threshold: int = 300,
    ) -> None:
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self._vad: _Optional[object] = None

        if _HAS_WEBRTCVAD:
            self._vad = _webrtcvad.Vad(aggressiveness)

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Return True if *audio_chunk* likely contains speech."""
        if self._vad is not None:
            try:
                return self._vad.is_speech(audio_chunk, self.sample_rate)  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("WebRTC VAD failed, falling back to energy detection: %s", e)
        return self._energy_detect(audio_chunk)

    def _energy_detect(self, audio_chunk: bytes) -> bool:
        """Simple RMS-energy detector as fallback."""
        n_samples = len(audio_chunk) // 2
        if n_samples == 0:
            return False
        samples = _struct.unpack(f"<{n_samples}h", audio_chunk)
        rms = (sum(s * s for s in samples) / n_samples) ** 0.5
        return rms > self.energy_threshold


def create_vad(engine: str = "silero", **kwargs):
    """Factory: silero -> webrtcvad -> energy-based."""
    if engine == "silero":
        try:
            from homie_core.voice.vad_silero import SileroVAD
            return SileroVAD(**kwargs)
        except ImportError:
            logger.warning("SileroVAD unavailable, falling back to webrtcvad")
            engine = "webrtcvad"
    if engine == "webrtcvad":
        if _HAS_WEBRTCVAD:
            return VAD(**kwargs)
        logger.warning("webrtcvad unavailable, falling back to energy-based VAD")
    return VoiceActivityDetector(**kwargs)
