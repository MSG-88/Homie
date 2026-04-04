from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from homie_core.plugins.base import HomiePlugin, PluginResult


class PluginManager:
    def __init__(self):
        self._plugins: dict[str, HomiePlugin] = {}
        self._enabled: set[str] = set()
        self._lock = threading.Lock()

    def register(self, plugin: HomiePlugin) -> None:
        with self._lock:
            self._plugins[plugin.name] = plugin

    def load_from_directory(self, directory: str | Path) -> int:
        directory = Path(directory)
        loaded = 0
        if not directory.exists():
            return 0
        for f in directory.glob("*_plugin.py"):
            try:
                spec = importlib.util.spec_from_file_location(f.stem, str(f))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and issubclass(attr, HomiePlugin)
                                and attr is not HomiePlugin and hasattr(attr, 'name') and attr.name):
                            plugin = attr()
                            self.register(plugin)
                            loaded += 1
            except Exception as e:
                logger.warning("Failed to load plugin from '%s': %s", f.name, e)
        return loaded

    def enable(self, name: str, config: dict | None = None) -> bool:
        with self._lock:
            plugin = self._plugins.get(name)
            if not plugin:
                return False
            try:
                plugin.on_activate(config or {})
                self._enabled.add(name)
                return True
            except Exception:
                return False

    def disable(self, name: str) -> bool:
        with self._lock:
            plugin = self._plugins.get(name)
            if not plugin or name not in self._enabled:
                return False
            try:
                plugin.on_deactivate()
                self._enabled.discard(name)
                return True
            except Exception:
                self._enabled.discard(name)
                return False

    def get_plugin(self, name: str) -> Optional[HomiePlugin]:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {"name": p.name, "description": p.description, "version": p.version,
             "enabled": p.name in self._enabled, "permissions": p.permissions}
            for p in self._plugins.values()
        ]

    def list_enabled(self) -> list[str]:
        return list(self._enabled)

    def query_plugin(self, name: str, intent: str, params: dict | None = None) -> PluginResult:
        plugin = self._plugins.get(name)
        if not plugin or name not in self._enabled:
            return PluginResult(success=False, error=f"Plugin '{name}' not found or not enabled")
        try:
            return plugin.on_query(intent, params or {})
        except Exception as e:
            return PluginResult(success=False, error=str(e))

    def action_plugin(self, name: str, action: str, params: dict | None = None) -> PluginResult:
        plugin = self._plugins.get(name)
        if not plugin or name not in self._enabled:
            return PluginResult(success=False, error=f"Plugin '{name}' not found or not enabled")
        try:
            return plugin.on_action(action, params or {})
        except Exception as e:
            return PluginResult(success=False, error=str(e))

    def collect_context(self) -> dict[str, Any]:
        context = {}
        for name in self._enabled:
            plugin = self._plugins.get(name)
            if plugin:
                try:
                    ctx = plugin.on_context()
                    if ctx:
                        context[name] = ctx
                except Exception as e:
                    logger.warning("Plugin '%s' context collection failed: %s", name, e)
        return context
