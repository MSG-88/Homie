from __future__ import annotations
import logging
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


class SileroVAD:
    """Neural VAD using Silero (~2MB, CPU). Hysteresis: trigger at threshold, release at threshold-0.15."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        if torch is None:
            raise ImportError("torch required for SileroVAD")
        self._threshold = threshold
        self._release_threshold = threshold - 0.15
        self._sample_rate = sample_rate
        self._triggered = False
        self._model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._model.eval()

    def is_speech(self, audio_chunk: bytes) -> bool:
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)
        prob = self._model(audio_tensor, self._sample_rate).item()
        if self._triggered:
            if prob < self._release_threshold:
                self._triggered = False
        else:
            if prob >= self._threshold:
                self._triggered = True
        return self._triggered

    def reset(self) -> None:
        self._triggered = False
        self._model.reset_states()

    @property
    def is_available(self) -> bool:
        return _HAS_TORCH
