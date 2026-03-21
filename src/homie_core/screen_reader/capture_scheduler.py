from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from homie_core.config import ScreenReaderConfig
from homie_core.screen_reader.window_tracker import WindowTracker, WindowInfo
from homie_core.screen_reader.pii_filter import PIIFilter

logger = logging.getLogger(__name__)

_ABSOLUTE_BLOCKLIST = frozenset({
    "1password", "keepassxc", "bitwarden", "lastpass", "dashlane",
    "incognito", "private browsing", "inprivate",
    "credential manager", "windows security",
    "online banking", "net banking",
})


def _is_absolutely_blocked(title: str, process: str) -> bool:
    combined = (title + " " + process).lower()
    return any(term in combined for term in _ABSOLUTE_BLOCKLIST)


@dataclass
class ScreenContext:
    window_title: str = ""
    process_name: str = ""
    ocr_text: str = ""
    visual_summary: str = ""
    timestamp: float = 0.0


class CaptureScheduler:
    def __init__(
        self,
        config: ScreenReaderConfig,
        window_tracker: WindowTracker | None = None,
        ocr_reader=None,
        visual_analyzer=None,
    ):
        self._config = config
        self._tracker = window_tracker or WindowTracker(blocklist=config.blocklist)
        self._ocr = ocr_reader
        self._visual = visual_analyzer
        self._last_window: WindowInfo | None = None
        self._last_t2: float = 0.0
        self._last_t3: float = 0.0
        self._context = ScreenContext()

    def tick(self) -> ScreenContext:
        if not self._config.enabled or self._config.dnd:
            return self._context

        now = time.time()
        window = self._tracker.get_active_window()

        if window is None:
            return self._context

        # Hard security check — block password managers, banking, and private windows
        # regardless of user configuration.
        if _is_absolutely_blocked(window.title, window.process_name):
            return self._context

        # T1: always track window
        window_changed = self._tracker.has_changed(self._last_window, window)
        self._context.window_title = window.title
        self._context.process_name = window.process_name
        self._context.timestamp = now

        if self._tracker.is_blocked(window.title):
            self._last_window = window
            return self._context

        # T2: OCR (level >= 2)
        if self._config.level >= 2 and self._ocr:
            should_ocr = (
                (window_changed and self._config.event_driven)
                or (now - self._last_t2 >= self._config.poll_interval_t2)
            )
            if should_ocr:
                img = self._ocr.capture_screen()
                if img:
                    text = self._ocr.extract_text(img)
                    self._context.ocr_text = text or ""
                    self._last_t2 = now

        # T3: Visual analysis (level >= 3)
        if self._config.level >= 3 and self._visual:
            should_analyze = (
                (window_changed and self._config.event_driven)
                or (now - self._last_t3 >= self._config.poll_interval_t3)
            )
            if should_analyze:
                img = self._ocr.capture_screen() if self._ocr else None
                if img:
                    summary = self._visual.analyze(img)
                    self._context.visual_summary = summary or ""
                    self._last_t3 = now

        self._last_window = window
        return self._context

    def get_context(self) -> ScreenContext:
        return self._context
