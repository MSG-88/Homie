from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginResult:
    success: bool
    data: Any = None
    error: str = ""


class HomiePlugin(ABC):
    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    permissions: list[str] = []

    @abstractmethod
    def on_activate(self, config: dict) -> None:
        pass

    @abstractmethod
    def on_deactivate(self) -> None:
        pass

    def on_context(self) -> dict[str, Any]:
        return {}

    def on_query(self, intent: str, params: dict) -> PluginResult:
        return PluginResult(success=False, error="Not implemented")

    def on_action(self, action: str, params: dict) -> PluginResult:
        return PluginResult(success=False, error="Not implemented")
