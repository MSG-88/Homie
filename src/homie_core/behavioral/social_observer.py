from __future__ import annotations

from collections import defaultdict
from typing import Any

from homie_core.behavioral.base import BaseObserver


class SocialObserver(BaseObserver):
    def __init__(self):
        super().__init__(name="social")
        self._contact_frequency: dict[str, int] = defaultdict(int)
        self._platform_usage: dict[str, int] = defaultdict(int)

    def tick(self) -> dict[str, Any]:
        return {}

    def record_communication(self, contact: str, platform: str) -> None:
        self._contact_frequency[contact] += 1
        self._platform_usage[platform] += 1
        self.record({"type": "communication", "contact": contact, "platform": platform})

    def get_profile_updates(self) -> dict[str, Any]:
        top_contacts = sorted(self._contact_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        top_platforms = sorted(self._platform_usage.items(), key=lambda x: x[1], reverse=True)
        return {
            "frequent_contacts": [c[0] for c in top_contacts],
            "preferred_platforms": [p[0] for p in top_platforms],
        }
