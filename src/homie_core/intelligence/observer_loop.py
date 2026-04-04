from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from homie_core.context.screen_monitor import ScreenMonitor, WindowInfo
from homie_core.context.app_tracker import AppTracker
from homie_core.intelligence.task_graph import TaskGraph
from homie_core.memory.working import WorkingMemory
from homie_core.utils import utc_now

if TYPE_CHECKING:
    from homie_core.neural.context_engine import SemanticContextEngine
    from homie_core.neural.activity_classifier import ActivityClassifier
    from homie_core.neural.rhythm_model import CircadianRhythmModel
    from homie_core.neural.behavioral_profile import BehavioralProfile
    from homie_core.neural.preference_engine import PreferenceEngine
    from homie_core.intelligence.workflow_predictor import WorkflowPredictor
    from homie_core.intelligence.flow_detector import FlowDetector
    from homie_core.intelligence.proactive_engine import ProactiveEngine


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
        rhythm_model: Optional[CircadianRhythmModel] = None,
        behavioral_profile: Optional[BehavioralProfile] = None,
        preference_engine: Optional[PreferenceEngine] = None,
        workflow_predictor: Optional[WorkflowPredictor] = None,
        flow_detector: Optional[FlowDetector] = None,
        proactive_engine: Optional[ProactiveEngine] = None,
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
        self._rhythm = rhythm_model
        self._profile = behavioral_profile
        self._prefs = preference_engine
        self._workflow = workflow_predictor
        self._flow = flow_detector
        self._proactive = proactive_engine
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_window: Optional[WindowInfo] = None
        self._poll_count: int = 0

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

        # Phase 1: Neural components (optional)
        context_shifted = False
        if self._context_engine:
            self._context_engine.update(window.process_name, window.title)
            context_shifted = self._context_engine.detect_context_shift()
            self._wm.update("context_shift", context_shifted)

        top = None
        top_confidence = 0.0
        if self._activity_classifier:
            scores = self._activity_classifier.classify(window.process_name, window.title)
            top = max(scores, key=scores.get)
            top_confidence = scores.get(top, 0.0)
            self._wm.update("activity_type", top)
            self._wm.update("activity_scores", scores)
            self._wm.update("activity_confidence", top_confidence)

        # Phase 2: Personal Neural Profile
        if self._rhythm:
            from datetime import datetime
            hour = datetime.now().hour
            flow_score = self._wm.get("flow_score", 0.5)
            self._rhythm.record_activity(
                hour=hour,
                productivity_score=flow_score,
                activity_type=top,
            )
            # Write rhythmic prediction for current hour
            rhythmic_score = self._rhythm.predict_productivity(hour)
            self._wm.update("rhythmic_score", rhythmic_score)

        if self._profile and self._context_engine:
            vec = self._context_engine.get_context_vector()
            if any(v != 0.0 for v in vec):
                self._profile.observe(vec)

        if self._prefs and top:
            self._prefs.record("activity", top, scores.get(top, 0.5))
            self._prefs.record("tool", window.process_name, 1.0)
            # Write dominant preferences to working memory
            dominant = self._prefs.get_dominant("activity")
            if dominant:
                self._wm.update("preferred_activity", dominant)
            shifts = self._prefs.get_detected_shifts()
            if shifts:
                self._wm.update("preference_shifts", shifts[-3:])

        # Phase 3: Predictive Intelligence
        if self._workflow:
            activity = self._wm.get("activity_type", "unknown")
            self._workflow.observe(activity)
            predictions = self._workflow.predict_next(activity, top_n=1)
            if predictions:
                self._wm.update("predicted_next_activity", predictions[0][0])

        if self._flow:
            activity = self._wm.get("activity_type", "unknown")
            self._flow.record_activity(activity)
            self._wm.update("flow_score", self._flow.get_flow_score())
            self._wm.update("in_flow", self._flow.is_in_flow())

        # Phase 4: Task duration tracking
        self._update_task_duration()

    def _update_task_duration(self) -> None:
        """Track how long the user has been on the current task."""
        active_tasks = [t for t in self._tg.get_tasks() if t.state == "active"]
        if active_tasks:
            current = active_tasks[-1]
            self._wm.update("minutes_in_task", current.duration_minutes())
            # Build description from project/apps
            project = self._tg._extract_project_from_task(current)
            apps = ", ".join(sorted(current.apps))
            desc = f"{project} ({apps})" if project else apps
            self._wm.update("task_description", desc)
        else:
            self._wm.update("minutes_in_task", 0.0)
            self._wm.update("task_description", "")

    def _loop(self) -> None:
        while self._running:
            start = time.monotonic()
            try:
                window = self._screen.get_active_window()
                if self._screen.has_changed(window):
                    self._handle_window_change(window)

                self._wm.update("is_deep_work", self._apps.is_deep_work())
                self._wm.update("switch_count_30m", self._apps.get_switch_count(30))

                # Refresh task duration every poll cycle (not just on window change)
                self._update_task_duration()

                # Proactive suggestions — the engine manages its own timer
                # internally so calling tick() every poll cycle is safe.
                if self._proactive:
                    self._proactive.tick()
            except Exception as e:
                logger.warning("Observer loop poll cycle error: %s", e)

            elapsed = time.monotonic() - start
            if elapsed > self._poll_interval * self._cpu_budget:
                self._poll_interval = min(self._poll_interval * 1.5, 10.0)

            self._poll_count += 1
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
