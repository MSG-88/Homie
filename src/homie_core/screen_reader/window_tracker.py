from __future__ import annotations
import fnmatch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

win32gui = None  # type: ignore[assignment]
psutil = None    # type: ignore[assignment]

try:
    import win32gui  # type: ignore[no-redef]
    import psutil    # type: ignore[no-redef]
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False


@dataclass
class WindowInfo:
    title: str
    process_name: str
    pid: int


class WindowTracker:
    def __init__(self, blocklist: list[str] | None = None):
        self._blocklist = [p.lower() for p in (blocklist or [])]

    def get_active_window(self) -> WindowInfo | None:
        if win32gui is None or psutil is None:
            return None
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32gui.GetWindowThreadProcessId(hwnd)
            proc_name = psutil.Process(pid).name()
            return WindowInfo(title=title, process_name=proc_name, pid=pid)
        except Exception:
            logger.debug("Failed to get active window", exc_info=True)
            return None

    def has_changed(self, old: WindowInfo | None, new: WindowInfo | None) -> bool:
        if old is None or new is None:
            return old is not new
        return old.title != new.title or old.process_name != new.process_name

    def is_blocked(self, title: str) -> bool:
        title_lower = title.lower()
        return any(fnmatch.fnmatch(title_lower, pattern) for pattern in self._blocklist)
