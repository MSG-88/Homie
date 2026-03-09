from __future__ import annotations

import threading
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from homie_core.plugins.base import PluginResult


class PluginSandbox:
    def __init__(self, timeout_seconds: float = 30.0, max_workers: int = 4):
        self._timeout = timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="plugin")
        self._crash_log: list[dict] = []

    def execute(self, func: Callable, *args, **kwargs) -> PluginResult:
        try:
            future = self._executor.submit(func, *args, **kwargs)
            result = future.result(timeout=self._timeout)
            if isinstance(result, PluginResult):
                return result
            return PluginResult(success=True, data=result)
        except FutureTimeout:
            self._crash_log.append({"error": "timeout", "func": str(func)})
            return PluginResult(success=False, error=f"Plugin timed out after {self._timeout}s")
        except Exception as e:
            self._crash_log.append({"error": str(e), "func": str(func)})
            return PluginResult(success=False, error=f"Plugin crashed: {e}")

    def get_crash_log(self) -> list[dict]:
        return list(self._crash_log)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
