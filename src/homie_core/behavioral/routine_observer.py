from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from homie_core.behavioral.base import BaseObserver
from homie_core.utils import utc_now


class RoutineObserver(BaseObserver):
    def __init__(self):
        super().__init__(name="routine")
        self._hourly_activity: dict[int, int] = defaultdict(int)
        self._daily_first_seen: dict[str, str] = {}
        self._daily_last_seen: dict[str, str] = {}
        self._break_gaps: list[float] = []
        self._last_activity: datetime | None = None

    def tick(self) -> dict[str, Any]:
        now = utc_now()
        today = now.strftime("%Y-%m-%d")
        hour = now.hour
        self._hourly_activity[hour] += 1
        if today not in self._daily_first_seen:
            self._daily_first_seen[today] = now.isoformat()
        self._daily_last_seen[today] = now.isoformat()

        if self._last_activity:
            gap_minutes = (now - self._last_activity).total_seconds() / 60
            if gap_minutes > 5:  # break detected
                self._break_gaps.append(gap_minutes)
                self.record({"type": "break", "duration_minutes": round(gap_minutes, 1)})
        self._last_activity = now
        return {"hour": hour, "active": True}

    def get_profile_updates(self) -> dict[str, Any]:
        peak_hours = sorted(self._hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        avg_break = sum(self._break_gaps) / max(1, len(self._break_gaps))
        return {
            "peak_hours": [h[0] for h in peak_hours],
            "average_break_minutes": round(avg_break, 1),
            "days_tracked": len(self._daily_first_seen),
        }
