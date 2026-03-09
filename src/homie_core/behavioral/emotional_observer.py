from __future__ import annotations

from collections import deque
from typing import Any

from homie_core.behavioral.base import BaseObserver
from homie_core.utils import utc_now


class EmotionalObserver(BaseObserver):
    def __init__(self):
        super().__init__(name="emotional")
        self._app_switches = deque(maxlen=100)
        self._typing_speeds: deque = deque(maxlen=50)
        self._current_mood: str = "neutral"

    def tick(self) -> dict[str, Any]:
        return {"mood": self._current_mood}

    def record_app_switch(self) -> None:
        self._app_switches.append(utc_now().isoformat())
        self._assess_mood()

    def record_typing_speed(self, chars_per_minute: float) -> None:
        self._typing_speeds.append(chars_per_minute)
        self._assess_mood()

    def _assess_mood(self) -> None:
        # High app switching = potential frustration
        recent_switches = len([s for s in self._app_switches
                               if s > (utc_now().isoformat()[:16])])  # rough same-minute check
        if recent_switches > 10:
            self._current_mood = "frustrated"
        elif recent_switches < 2 and self._typing_speeds:
            avg_speed = sum(self._typing_speeds) / len(self._typing_speeds)
            if avg_speed > 200:
                self._current_mood = "flow"
            else:
                self._current_mood = "focused"
        else:
            self._current_mood = "neutral"

    def get_profile_updates(self) -> dict[str, Any]:
        return {
            "current_mood": self._current_mood,
            "mood_assessment_method": "behavioral_signals",
        }
