"""Proactive Intelligence Module — anticipates user needs instead of just reacting.

Provides:
- Morning briefing generator: overnight emails, today's calendar, pending tasks,
  knowledge graph trending topics.
- Follow-up tracker: captures "I'll do X later" / "remind me about Y" commitments
  and surfaces them at appropriate times.
- Context-aware suggestions: time-of-day and activity-based action suggestions.
- Pattern detector: recognises recurring behavioural patterns and preemptively
  prepares relevant context.

Designed to run as a background task inside the daemon, producing non-intrusive
desktop notifications and data for the /briefing and /suggestions commands.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from homie_core.intelligence.commitment_extractor import Commitment, extract_commitments

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FollowUp:
    """A tracked follow-up extracted from conversation."""
    id: str
    text: str
    due_by: Optional[str] = None      # raw human string like "tomorrow"
    created_at: float = 0.0
    source: str = "conversation"
    surfaced: bool = False
    surfaced_at: float = 0.0
    dismissed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FollowUp":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BriefingData:
    """Structured morning briefing payload."""
    generated_at: str
    greeting: str
    email_summary: str = ""
    calendar_summary: str = ""
    pending_followups: list[dict] = field(default_factory=list)
    trending_topics: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def format_text(self) -> str:
        lines = [self.greeting, ""]
        if self.email_summary:
            lines.append("  Email:")
            lines.append(f"    {self.email_summary}")
            lines.append("")
        if self.calendar_summary:
            lines.append("  Calendar:")
            lines.append(f"    {self.calendar_summary}")
            lines.append("")
        if self.pending_followups:
            lines.append("  Follow-ups:")
            for fu in self.pending_followups:
                due = f" (due: {fu.get('due_by', 'unset')})" if fu.get("due_by") else ""
                lines.append(f"    - {fu['text']}{due}")
            lines.append("")
        if self.trending_topics:
            lines.append("  Trending in your knowledge graph:")
            for topic in self.trending_topics[:5]:
                lines.append(f"    - {topic}")
            lines.append("")
        if self.suggestions:
            lines.append("  Suggestions:")
            for s in self.suggestions:
                lines.append(f"    - {s}")
            lines.append("")
        if len(lines) <= 2:
            lines.append("  No pending items. Fresh start today!")
        return "\n".join(lines)


@dataclass
class PatternRecord:
    """A detected behavioural pattern."""
    action: str
    typical_hour: int
    occurrences: int = 0
    last_seen: float = 0.0


# ---------------------------------------------------------------------------
# Follow-up Tracker (persistent)
# ---------------------------------------------------------------------------

class FollowUpTracker:
    """Captures user commitments and resurfaces them at appropriate times.

    Persists to ``~/.homie/intelligence/followups.json``.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._dir = storage_dir or (Path.home() / ".homie" / "intelligence")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "followups.json"
        self._items: list[FollowUp] = []
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._items = [FollowUp.from_dict(d) for d in data]
            except Exception:
                self._items = []

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps([f.to_dict() for f in self._items], indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save follow-ups: %s", exc)

    # -- public API ----------------------------------------------------------

    def ingest(self, text: str, source: str = "conversation") -> list[FollowUp]:
        """Scan *text* for commitments and track them. Returns newly added items."""
        commitments = extract_commitments(text, source=source)
        added: list[FollowUp] = []
        for c in commitments:
            # Deduplicate by text similarity
            if any(existing.text.lower() == c.text.lower() for existing in self._items):
                continue
            fu = FollowUp(
                id=f"fu_{int(time.time() * 1000)}_{len(self._items)}",
                text=c.text,
                due_by=c.due_by,
                created_at=time.time(),
                source=source,
            )
            self._items.append(fu)
            added.append(fu)
        if added:
            self._save()
        return added

    def get_pending(self) -> list[FollowUp]:
        """Return all non-dismissed, non-surfaced follow-ups."""
        return [f for f in self._items if not f.dismissed and not f.surfaced]

    def get_due(self) -> list[FollowUp]:
        """Return follow-ups that are due based on simple heuristics.

        Items with no due_by are surfaced after 4 hours.
        Items with due_by containing 'today', 'now', 'morning', 'tonight'
        are surfaced immediately.
        Items with 'tomorrow' are surfaced the next day.
        Everything else is surfaced after 24 hours as a fallback.
        """
        now = time.time()
        due: list[FollowUp] = []
        for f in self._items:
            if f.dismissed or f.surfaced:
                continue
            age_hours = (now - f.created_at) / 3600

            if f.due_by:
                lower = f.due_by.lower()
                if any(kw in lower for kw in ("now", "today", "morning", "tonight", "asap")):
                    due.append(f)
                elif "tomorrow" in lower and age_hours >= 16:
                    due.append(f)
                elif age_hours >= 24:
                    due.append(f)
            else:
                # No explicit deadline — surface after 4 hours
                if age_hours >= 4:
                    due.append(f)
        return due

    def mark_surfaced(self, follow_up_id: str) -> None:
        for f in self._items:
            if f.id == follow_up_id:
                f.surfaced = True
                f.surfaced_at = time.time()
                break
        self._save()

    def dismiss(self, follow_up_id: str) -> None:
        for f in self._items:
            if f.id == follow_up_id:
                f.dismissed = True
                break
        self._save()

    def list_all(self) -> list[FollowUp]:
        return list(self._items)

    def cleanup(self, max_age_days: int = 30) -> int:
        """Remove dismissed/old items. Returns count removed."""
        cutoff = time.time() - max_age_days * 86400
        before = len(self._items)
        self._items = [
            f for f in self._items
            if not (f.dismissed and f.created_at < cutoff)
        ]
        removed = before - len(self._items)
        if removed:
            self._save()
        return removed


# ---------------------------------------------------------------------------
# Pattern Detector
# ---------------------------------------------------------------------------

class PatternDetector:
    """Detects recurring behavioural patterns from working memory snapshots.

    Tracks what actions happen at what hours and builds a frequency table.
    After enough observations, can predict what the user typically does at a
    given time.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._dir = storage_dir or (Path.home() / ".homie" / "intelligence")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "patterns.json"
        # {action: {hour_str: count}}
        self._freq: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._freq = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._freq = {}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._freq, indent=2), encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save patterns: %s", exc)

    def observe(self, action: str, hour: Optional[int] = None) -> None:
        """Record an observation of *action* at *hour* (default: current hour)."""
        if hour is None:
            hour = datetime.now().hour
        h = str(hour)
        if action not in self._freq:
            self._freq[action] = {}
        self._freq[action][h] = self._freq[action].get(h, 0) + 1
        self._save()

    def predict(self, hour: Optional[int] = None, top_n: int = 3) -> list[tuple[str, int]]:
        """Return top-N actions predicted for *hour*, with their counts."""
        if hour is None:
            hour = datetime.now().hour
        h = str(hour)
        candidates: list[tuple[str, int]] = []
        for action, hours in self._freq.items():
            count = hours.get(h, 0)
            if count >= 2:  # need at least 2 observations
                candidates.append((action, count))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def get_patterns(self, min_occurrences: int = 3) -> list[PatternRecord]:
        """Return all detected patterns with at least *min_occurrences*."""
        patterns: list[PatternRecord] = []
        for action, hours in self._freq.items():
            for h, count in hours.items():
                if count >= min_occurrences:
                    patterns.append(PatternRecord(
                        action=action,
                        typical_hour=int(h),
                        occurrences=count,
                    ))
        patterns.sort(key=lambda p: p.occurrences, reverse=True)
        return patterns


# ---------------------------------------------------------------------------
# Proactive Intelligence Coordinator
# ---------------------------------------------------------------------------

class ProactiveIntelligence:
    """Central coordinator for proactive intelligence features.

    Designed to be instantiated once by the daemon and ticked periodically.
    Produces :class:`BriefingData` and notification payloads without directly
    touching the UI — the daemon routes them to the notification system.
    """

    def __init__(
        self,
        working_memory=None,
        email_service=None,
        calendar_provider=None,
        knowledge_graph=None,
        session_tracker=None,
        user_name: str = "",
        storage_dir: Optional[Path] = None,
    ):
        self._wm = working_memory
        self._email = email_service
        self._calendar = calendar_provider
        self._kg = knowledge_graph
        self._session_tracker = session_tracker
        self._user_name = user_name

        base_dir = storage_dir or (Path.home() / ".homie" / "intelligence")
        self._followups = FollowUpTracker(storage_dir=base_dir)
        self._patterns = PatternDetector(storage_dir=base_dir)

        self._last_briefing_date: str = ""
        self._last_suggestion_time: float = 0.0
        self._suggestion_cooldown: float = 300.0  # 5 min

    # -- accessors -----------------------------------------------------------

    @property
    def followup_tracker(self) -> FollowUpTracker:
        return self._followups

    @property
    def pattern_detector(self) -> PatternDetector:
        return self._patterns

    # -- morning briefing ----------------------------------------------------

    def generate_briefing(self) -> BriefingData:
        """Build a comprehensive morning briefing from all data sources."""
        now = datetime.now()
        hour = now.hour
        if hour < 12:
            tod = "morning"
        elif hour < 17:
            tod = "afternoon"
        else:
            tod = "evening"

        name_part = f", {self._user_name}" if self._user_name else ""
        greeting = f"Good {tod}{name_part}! Here is your briefing for {now.strftime('%A, %B %d')}:"

        email_summary = self._gather_email_summary()
        calendar_summary = self._gather_calendar_summary()
        pending = [f.to_dict() for f in self._followups.get_pending()[:10]]
        topics = self._gather_trending_topics()
        suggestions = self._generate_suggestions()

        self._last_briefing_date = now.strftime("%Y-%m-%d")

        return BriefingData(
            generated_at=now.isoformat(),
            greeting=greeting,
            email_summary=email_summary,
            calendar_summary=calendar_summary,
            pending_followups=pending,
            trending_topics=topics,
            suggestions=suggestions,
        )

    def should_fire_briefing(self) -> bool:
        """True if morning briefing hasn't been delivered today and it's 6-10am."""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if today == self._last_briefing_date:
            return False
        return 6 <= now.hour < 10

    # -- follow-up surfacing -------------------------------------------------

    def check_followups(self) -> list[FollowUp]:
        """Return follow-ups that should be surfaced now."""
        due = self._followups.get_due()
        for fu in due:
            self._followups.mark_surfaced(fu.id)
        return due

    def ingest_conversation(self, text: str, source: str = "conversation") -> list[FollowUp]:
        """Scan a user message for commitments and track them."""
        return self._followups.ingest(text, source=source)

    # -- context-aware suggestions -------------------------------------------

    def get_suggestions(self) -> list[str]:
        """Generate on-demand context-aware suggestions."""
        return self._generate_suggestions()

    # -- daemon tick ---------------------------------------------------------

    def tick(self) -> dict[str, Any]:
        """Called periodically from the daemon loop.

        Returns a dict with optional keys:
          - "briefing": BriefingData if morning briefing should fire
          - "followups": list[FollowUp] if any are due
          - "pattern_actions": list of predicted actions for current hour
        """
        result: dict[str, Any] = {}

        # Morning briefing
        if self.should_fire_briefing():
            result["briefing"] = self.generate_briefing()

        # Follow-up surfacing
        due_followups = self.check_followups()
        if due_followups:
            result["followups"] = due_followups

        # Pattern observation: record current activity
        if self._wm:
            activity = self._wm.get("activity_type", "unknown")
            if activity and activity != "unknown":
                self._patterns.observe(activity)

        return result

    # -- private helpers -----------------------------------------------------

    def _gather_email_summary(self) -> str:
        """Pull email summary from the email service if available."""
        if not self._email:
            return ""
        try:
            summary = self._email.get_summary(days=1)
            unread = summary.get("unread", 0)
            hp = summary.get("high_priority", [])
            if unread == 0:
                return "Inbox clear — no unread emails overnight."
            parts = [f"{unread} unread email{'s' if unread != 1 else ''}"]
            if hp:
                parts.append(f"{len(hp)} high priority")
                top_subjects = [m.get("subject", "(no subject)") for m in hp[:3]]
                parts.append("Top: " + "; ".join(top_subjects))
            return ". ".join(parts) + "."
        except Exception as exc:
            logger.debug("Email summary failed: %s", exc)
            return ""

    def _gather_calendar_summary(self) -> str:
        """Pull today's events from the calendar provider if available."""
        if not self._calendar:
            return ""
        try:
            if not getattr(self._calendar, "_connected", False):
                return ""
            now = datetime.now(timezone.utc)
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            events = self._calendar.list_events(
                time_min=start_of_day, time_max=end_of_day, max_results=10,
            )
            if not events:
                return "No events scheduled for today."
            briefs = [e.to_brief() if hasattr(e, "to_brief") else str(e) for e in events]
            return f"{len(events)} event{'s' if len(events) != 1 else ''} today: " + " | ".join(briefs)
        except Exception as exc:
            logger.debug("Calendar summary failed: %s", exc)
            return ""

    def _gather_trending_topics(self) -> list[str]:
        """Pull recent/trending topics from the knowledge graph."""
        if not self._kg:
            return []
        try:
            if hasattr(self._kg, "get_recent_facts"):
                facts = self._kg.get_recent_facts(limit=5)
                return [f.get("fact", str(f)) for f in facts]
            if hasattr(self._kg, "get_facts"):
                facts = self._kg.get_facts(limit=5)
                return [f.get("fact", str(f)) if isinstance(f, dict) else str(f) for f in facts]
        except Exception as exc:
            logger.debug("Knowledge graph query failed: %s", exc)
        return []

    def _generate_suggestions(self) -> list[str]:
        """Build context-aware suggestions based on time, activity, patterns."""
        suggestions: list[str] = []
        now = datetime.now()
        hour = now.hour

        # Time-based suggestions
        if 6 <= hour <= 9:
            suggestions.append("Review overnight emails and plan your day")
        elif 12 <= hour <= 13:
            suggestions.append("Good time for a break or light reading")
        elif 17 <= hour <= 18:
            suggestions.append("Consider wrapping up — review today's progress")

        # Pattern-based suggestions
        predicted = self._patterns.predict(hour=hour, top_n=2)
        for action, count in predicted:
            suggestions.append(f"You usually do '{action}' around this time ({count}x observed)")

        # Follow-up reminders
        pending = self._followups.get_pending()
        if pending:
            suggestions.append(f"You have {len(pending)} pending follow-up{'s' if len(pending) != 1 else ''}")

        # Working memory signals
        if self._wm:
            mins = self._wm.get("minutes_in_task", 0)
            flow = self._wm.get("flow_score", 0.5)
            if mins >= 60 and flow < 0.4:
                suggestions.append("You seem stuck — consider taking a break or switching tasks")
            if flow >= 0.8 and mins >= 20:
                suggestions.append("You are in a good flow state — minimize distractions")

        return suggestions
