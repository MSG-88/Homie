from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PriorityItem:
    description: str
    score: float  # 0.0 to 1.0
    reason: str   # "stuck", "deadline", "recurrent", "commitment"
    source: str = ""

class PriorityInference:
    """Infers what matters most to the user right now."""

    def infer(
        self,
        commitments: list[dict] | None = None,
        incomplete_tasks: list[dict] | None = None,
        recent_topics: list[str] | None = None,
    ) -> list[PriorityItem]:
        priorities = []

        # Commitments with approaching deadlines
        for c in (commitments or []):
            due = c.get("due_by", "")
            score = 0.8
            if due and any(w in due.lower() for w in ["today", "now", "asap", "urgent"]):
                score = 0.95
            elif due and any(w in due.lower() for w in ["tomorrow", "morning"]):
                score = 0.85
            priorities.append(PriorityItem(
                description=c.get("text", ""),
                score=score,
                reason="commitment",
                source=c.get("source", ""),
            ))

        # Incomplete/stuck tasks
        for t in (incomplete_tasks or []):
            state = t.get("state", "active")
            score = 0.9 if state == "stuck" else 0.6
            priorities.append(PriorityItem(
                description=t.get("description", t.get("task", "")),
                score=score,
                reason="stuck" if state == "stuck" else "incomplete",
            ))

        # Recurrent topics (user keeps coming back to these)
        for i, topic in enumerate(recent_topics or []):
            score = max(0.3, 0.7 - i * 0.1)  # decreasing by rank
            priorities.append(PriorityItem(
                description=topic,
                score=score,
                reason="recurrent",
            ))

        return sorted(priorities, key=lambda p: p.score, reverse=True)
