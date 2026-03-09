from __future__ import annotations

from collections import defaultdict
from typing import Any


class HabitDetector:
    def __init__(self, min_occurrences: int = 3, min_confidence: float = 0.6):
        self._sequences: dict[str, list[dict]] = defaultdict(list)
        self._trigger_responses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._min_occurrences = min_occurrences
        self._min_confidence = min_confidence

    def record_action(self, action: str, context: dict | None = None) -> None:
        from homie_core.utils import utc_now
        entry = {"action": action, "context": context or {}, "timestamp": utc_now().isoformat()}
        hour = utc_now().hour
        day = utc_now().strftime("%A")
        time_key = f"{day}_{hour}"
        self._sequences[time_key].append(entry)

    def record_trigger_response(self, trigger: str, response: str) -> None:
        self._trigger_responses[trigger][response] += 1

    def detect_habits(self) -> list[dict[str, Any]]:
        habits = []
        # Time-based habits
        for time_key, actions in self._sequences.items():
            action_counts: dict[str, int] = defaultdict(int)
            for a in actions:
                action_counts[a["action"]] += 1
            for action, count in action_counts.items():
                if count >= self._min_occurrences:
                    habits.append({
                        "type": "time_based",
                        "time": time_key,
                        "action": action,
                        "count": count,
                        "confidence": min(1.0, count / (self._min_occurrences * 2)),
                    })
        # Trigger-response habits
        for trigger, responses in self._trigger_responses.items():
            total = sum(responses.values())
            for response, count in responses.items():
                confidence = count / total
                if count >= self._min_occurrences and confidence >= self._min_confidence:
                    habits.append({
                        "type": "trigger_response",
                        "trigger": trigger,
                        "response": response,
                        "count": count,
                        "confidence": round(confidence, 2),
                    })
        return habits

    def suggest_automations(self) -> list[dict]:
        habits = self.detect_habits()
        suggestions = []
        for h in habits:
            if h["confidence"] >= 0.8:
                if h["type"] == "trigger_response":
                    suggestions.append({
                        "description": f"Auto-{h['response']} when {h['trigger']}",
                        "trigger": h["trigger"],
                        "action": h["response"],
                        "confidence": h["confidence"],
                    })
        return suggestions
