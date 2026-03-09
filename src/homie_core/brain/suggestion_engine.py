from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from homie_core.utils import utc_now


class Suggestion:
    def __init__(self, text: str, category: str, confidence: float, source: str = ""):
        self.text = text
        self.category = category
        self.confidence = confidence
        self.source = source
        self.timestamp = utc_now().isoformat()


class SuggestionEngine:
    def __init__(self, threshold: float = 0.6):
        self._threshold = threshold
        self._history: list[dict] = []
        self._acceptance_by_category: dict[str, dict] = defaultdict(
            lambda: {"accepted": 0, "total": 0}
        )
        self._suppressed_until: Optional[str] = None

    def generate_suggestions(self, context: dict[str, Any], beliefs: list[dict] | None = None) -> list[Suggestion]:
        suggestions = []

        # Break reminder
        if context.get("is_deep_work") and not self._is_suppressed():
            suggestions.append(Suggestion(
                text="You've been focused for a while. Consider a short break.",
                category="health",
                confidence=0.7,
                source="deep_work_detection",
            ))

        # Meeting reminder (from context)
        if context.get("upcoming_meeting"):
            suggestions.append(Suggestion(
                text=f"Meeting in {context['upcoming_meeting']} minutes.",
                category="calendar",
                confidence=0.95,
                source="calendar_plugin",
            ))

        # Filter by threshold and category acceptance
        filtered = []
        for s in suggestions:
            cat_stats = self._acceptance_by_category[s.category]
            if cat_stats["total"] > 5:
                acceptance_rate = cat_stats["accepted"] / cat_stats["total"]
                if acceptance_rate < 0.2:
                    continue  # user almost never accepts these
            if s.confidence >= self._threshold:
                filtered.append(s)

        return filtered

    def record_response(self, category: str, accepted: bool) -> None:
        self._acceptance_by_category[category]["total"] += 1
        if accepted:
            self._acceptance_by_category[category]["accepted"] += 1
        self._history.append({
            "category": category,
            "accepted": accepted,
            "timestamp": utc_now().isoformat(),
        })

    def suppress(self, minutes: int = 30) -> None:
        from datetime import timedelta
        until = utc_now() + timedelta(minutes=minutes)
        self._suppressed_until = until.isoformat()

    def _is_suppressed(self) -> bool:
        if not self._suppressed_until:
            return False
        from datetime import datetime, timezone
        until = datetime.fromisoformat(self._suppressed_until)
        return utc_now() < until

    def get_acceptance_rates(self) -> dict[str, float]:
        rates = {}
        for cat, stats in self._acceptance_by_category.items():
            if stats["total"] > 0:
                rates[cat] = stats["accepted"] / stats["total"]
        return rates
