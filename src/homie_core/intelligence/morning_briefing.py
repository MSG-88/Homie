from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class BriefingSection:
    title: str
    items: list[str] = field(default_factory=list)

@dataclass
class MorningBriefing:
    greeting: str
    sections: list[BriefingSection] = field(default_factory=list)
    generated_at: str = ""

class MorningBriefingOrchestrator:
    """Generates a morning briefing from multiple data sources."""

    def __init__(self, earliest_hour: int = 6, latest_hour: int = 10):
        self._earliest = earliest_hour
        self._latest = latest_hour
        self._last_fired_date: str = ""

    def should_fire(self) -> bool:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if today == self._last_fired_date:
            return False
        return self._earliest <= now.hour < self._latest

    def generate(
        self,
        user_name: str = "",
        commitments: list[dict] | None = None,
        recent_facts: list[dict] | None = None,
        incomplete_tasks: list[dict] | None = None,
        session_summary: str | None = None,
    ) -> MorningBriefing:
        self._last_fired_date = datetime.now().strftime("%Y-%m-%d")
        hour = datetime.now().hour
        greeting = f"Good morning{', ' + user_name if user_name else ''}! Here's your briefing:"

        sections = []

        # Pending commitments
        if commitments:
            items = [f"{c.get('text', '')} (due: {c.get('due_by', 'unset')})" for c in commitments]
            sections.append(BriefingSection(title="Pending Commitments", items=items))

        # Incomplete tasks from last session
        if incomplete_tasks:
            items = [t.get("description", t.get("task", "")) for t in incomplete_tasks]
            sections.append(BriefingSection(title="Unfinished Tasks", items=items))

        # Yesterday's summary
        if session_summary:
            sections.append(BriefingSection(title="Yesterday's Summary", items=[session_summary]))

        # Key facts recently learned
        if recent_facts:
            items = [f.get("fact", "") for f in recent_facts[:5]]
            sections.append(BriefingSection(title="Recently Learned", items=items))

        return MorningBriefing(
            greeting=greeting,
            sections=sections,
            generated_at=datetime.now().isoformat(),
        )

    def format_text(self, briefing: MorningBriefing) -> str:
        """Format briefing as plain text for CLI display."""
        lines = [briefing.greeting, ""]
        for section in briefing.sections:
            lines.append(f"  {section.title}:")
            for item in section.items:
                lines.append(f"    - {item}")
            lines.append("")
        if not briefing.sections:
            lines.append("  No pending items. Fresh start today!")
        return "\n".join(lines)
