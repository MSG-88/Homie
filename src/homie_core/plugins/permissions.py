from __future__ import annotations

from typing import Any


class PermissionManager:
    def __init__(self):
        self._granted: dict[str, set[str]] = {}  # plugin_name -> set of granted permissions

    def request_permissions(self, plugin_name: str, requested: list[str]) -> list[str]:
        return requested  # In real app, this would prompt the user

    def grant(self, plugin_name: str, permissions: list[str]) -> None:
        if plugin_name not in self._granted:
            self._granted[plugin_name] = set()
        self._granted[plugin_name].update(permissions)

    def revoke(self, plugin_name: str, permission: str) -> None:
        if plugin_name in self._granted:
            self._granted[plugin_name].discard(permission)

    def revoke_all(self, plugin_name: str) -> None:
        self._granted.pop(plugin_name, None)

    def has_permission(self, plugin_name: str, permission: str) -> bool:
        return permission in self._granted.get(plugin_name, set())

    def get_granted(self, plugin_name: str) -> list[str]:
        return list(self._granted.get(plugin_name, set()))

    def check_all(self, plugin_name: str, required: list[str]) -> bool:
        granted = self._granted.get(plugin_name, set())
        return all(p in granted for p in required)
