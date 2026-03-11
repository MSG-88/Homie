from __future__ import annotations

import threading
import time
from typing import Callable, Optional, TYPE_CHECKING

from homie_core.context.screen_monitor import ScreenMonitor, WindowInfo
from homie_core.context.app_tracker import AppTracker
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.memory.working import WorkingMemory
from homie_core.utils import utc_now

if TYPE_CHECKING:
    from homie_core.neural.context_engine import SemanticContextEngine
    from homie_core.neural.activity_classifier import ActivityClassifier


class ObserverLoop:
    """Event-driven observer thread that watches OS state changes."""

    def __init__(
        self,
        working_memory: WorkingMemory,
        task_graph: TaskGraph,
        app_tracker: Optional[AppTracker] = None,
        screen_monitor: Optional[ScreenMonitor] = None,
        on_context_change: Optional[Callable[[str, str], None]] = None,
        poll_interval: float = 1.0,
        cpu_budget: float = 0.05,
        context_engine: Optional[SemanticContextEngine] = None,
        activity_classifier: Optional[ActivityClassifier] = None,
    ):
        self._wm = working_memory
        self._tg = task_graph
        self._apps = app_tracker or AppTracker()
        self._screen = screen_monitor or ScreenMonitor()
        self._on_context_change = on_context_change
        self._poll_interval = poll_interval
        self._cpu_budget = cpu_budget
        self._context_engine = context_engine
        self._activity_classifier = activity_classifier
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_window: Optional[WindowInfo] = None

    def _handle_window_change(self, window: WindowInfo) -> None:
        if (self._last_window and
            self._last_window.title == window.title and
            self._last_window.process_name == window.process_name):
            return
        self._last_window = window

        self._wm.update("active_window", window.title)
        self._wm.update("active_process", window.process_name)

        self._apps.track(window.process_name)

        self._tg.observe(
            process=window.process_name,
            title=window.title,
            timestamp=window.timestamp or utc_now().isoformat(),
        )

        if self._on_context_change:
            self._on_context_change(window.process_name, window.title)

        # Neural components (optional)
        if self._context_engine:
            self._context_engine.update(window.process_name, window.title)
        if self._activity_classifier:
            scores = self._activity_classifier.classify(window.process_name, window.title)
            top = max(scores, key=scores.get)
            self._wm.update("activity_type", top)
            self._wm.update("activity_scores", scores)

    def _loop(self) -> None:
        while self._running:
            start = time.monotonic()
            try:
                window = self._screen.get_active_window()
                if self._screen.has_changed(window):
                    self._handle_window_change(window)

                self._wm.update("is_deep_work", self._apps.is_deep_work())
                self._wm.update("switch_count_30m", self._apps.get_switch_count(30))
            except Exception:
                pass

            elapsed = time.monotonic() - start
            if elapsed > self._poll_interval * self._cpu_budget:
                self._poll_interval = min(self._poll_interval * 1.5, 10.0)

            time.sleep(self._poll_interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="observer")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def get_app_tracker(self) -> AppTracker:
        return self._apps
