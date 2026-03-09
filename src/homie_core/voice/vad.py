from __future__ import annotations

from collections import deque


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
