from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Optional

from homie_core.utils import utc_now


@dataclass
class WindowInfo:
    title: str = ""
    process_name: str = ""
    pid: int = 0
    timestamp: str = ""


class ScreenMonitor:
    def __init__(self):
        import warnings
        warnings.warn(
            "ScreenMonitor is deprecated. Use homie_core.screen_reader.window_tracker.WindowTracker instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._last_window: Optional[WindowInfo] = None

    def get_active_window(self) -> WindowInfo:
        if platform.system() == "Windows":
            return self._get_windows_active_window()
        return WindowInfo(title="unknown", process_name="unknown", timestamp=utc_now().isoformat())

    def _get_windows_active_window(self) -> WindowInfo:
        try:
            import win32gui
            import win32process
            import psutil

            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc_name = "unknown"

            info = WindowInfo(title=title, process_name=proc_name, pid=pid, timestamp=utc_now().isoformat())
            self._last_window = info
            return info
        except ImportError:
            return WindowInfo(title="unknown", process_name="unknown", timestamp=utc_now().isoformat())
        except Exception:
            if self._last_window:
                return self._last_window
            return WindowInfo(title="unknown", process_name="unknown", timestamp=utc_now().isoformat())

    def has_changed(self, new_window: WindowInfo) -> bool:
        if self._last_window is None:
            return True
        return (self._last_window.title != new_window.title or
                self._last_window.process_name != new_window.process_name)
