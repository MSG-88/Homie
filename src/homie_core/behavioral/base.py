from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseObserver(ABC):
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._observations: list[dict] = []

    @abstractmethod
    def tick(self) -> dict[str, Any]:
        """Called periodically to collect new observations."""
        pass

    @abstractmethod
    def get_profile_updates(self) -> dict[str, Any]:
        """Return aggregated profile data from observations."""
        pass

    def get_observations(self) -> list[dict]:
        return list(self._observations)

    def clear_observations(self) -> None:
        self._observations.clear()

    def record(self, observation: dict) -> None:
        from homie_core.utils import utc_now
        observation["timestamp"] = utc_now().isoformat()
        observation["observer"] = self.name
        self._observations.append(observation)
