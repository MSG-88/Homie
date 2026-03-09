from __future__ import annotations

from typing import Any

from homie_core.behavioral.base import BaseObserver
from homie_core.memory.semantic import SemanticMemory


class ProfileSynthesizer:
    def __init__(self, semantic_memory: SemanticMemory, observers: list[BaseObserver] | None = None):
        self._sm = semantic_memory
        self._observers = observers or []

    def add_observer(self, observer: BaseObserver) -> None:
        self._observers.append(observer)

    def synthesize(self) -> dict[str, Any]:
        full_profile = {}
        for observer in self._observers:
            if observer.enabled:
                updates = observer.get_profile_updates()
                full_profile[observer.name] = updates
                self._sm.set_profile(observer.name, updates)
        return full_profile

    def get_full_profile(self) -> dict[str, Any]:
        return self._sm.get_all_profiles()
