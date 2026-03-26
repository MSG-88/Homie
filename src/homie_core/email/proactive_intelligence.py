"""Proactive Email Intelligence — automatic email analysis and insight surfacing."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmailInsight:
    """A single actionable insight from email analysis."""
    category: str       # "action_required", "deadline", "waiting_response", "important_update", "opportunity", "financial"
    priority: str       # "urgent", "high", "medium", "low"
    subject: str        # Brief description
    detail: str         # Full insight text
    source_email_id: str
    source_subject: str
    source_sender: str
    suggested_action: str = ""  # What Homie suggests doing
    deadline: str = ""          # If time-sensitive


@dataclass
class DailyBriefing:
    """Proactive daily briefing compiled from email analysis."""
    timestamp: float
    total_unread: int
    high_priority_count: int
    insights: list[EmailInsight] = field(default_factory=list)
    summary: str = ""           # Natural language overview
    suggested_focus: str = ""   # What to prioritize today

    def to_prompt_context(self) -> str:
        """Format as context for the system prompt."""
        if not self.insights and self.total_unread == 0:
            return ""

        lines = [f"\U0001f4e7 Email Briefing ({self.total_unread} unread, {self.high_priority_count} high-priority):"]

        if self.summary:
            lines.append(f"  {self.summary}")

        # Group by category
        urgent = [i for i in self.insights if i.priority == "urgent"]
        actions = [i for i in self.insights if i.category == "action_required" and i.priority != "urgent"]
        deadlines = [i for i in self.insights if i.category == "deadline"]
        waiting = [i for i in self.insights if i.category == "waiting_response"]

        if urgent:
            lines.append(f"  \U0001f534 URGENT ({len(urgent)}):")
            for i in urgent:
                lines.append(f"    - {i.subject} (from {i.source_sender})")
                if i.suggested_action:
                    lines.append(f"      \u2192 {i.suggested_action}")

        if actions:
            lines.append(f"  \U0001f7e1 Action needed ({len(actions)}):")
            for i in actions[:5]:
                lines.append(f"    - {i.subject}")

        if deadlines:
            lines.append(f"  \u23f0 Deadlines ({len(deadlines)}):")
            for i in deadlines[:3]:
                lines.append(f"    - {i.subject}: {i.deadline}")

        if waiting:
            lines.append(f"  \u23f3 Awaiting response ({len(waiting)}):")
            for i in waiting[:3]:
                lines.append(f"    - {i.subject}")

        if self.suggested_focus:
            lines.append(f"  \U0001f4a1 Suggested focus: {self.suggested_focus}")

        return "\n".join(lines)


class ProactiveEmailIntelligence:
    """Analyzes emails and generates proactive insights and briefings.

    This runs in the background after each email sync cycle, analyzing
    new and unread emails to surface actionable information.
    """

    def __init__(self, email_service, inference_fn: Optional[Callable] = None):
        """
        Args:
            email_service: The EmailService instance (has provider, classifier, cache access)
            inference_fn: Optional LLM inference function for deep analysis.
                         Called as inference_fn(prompt=..., max_tokens=..., temperature=...)
        """
        self._email = email_service
        self._inference_fn = inference_fn
        self._last_briefing: Optional[DailyBriefing] = None
        self._last_briefing_time: float = 0
        self._briefing_interval = 1800  # 30 min cache

    def generate_briefing(self, force: bool = False) -> DailyBriefing:
        """Generate a proactive briefing from current email state.

        Caches results for briefing_interval seconds unless force=True.
        """
        now = time.time()
        if not force and self._last_briefing and (now - self._last_briefing_time) < self._briefing_interval:
            return self._last_briefing

        # Gather email data
        insights = []
        total_unread = 0
        high_priority = 0

        try:
            # Get unread emails from cache
            unread = self._get_unread_emails()
            total_unread = len(unread)

            # Analyze each unread email
            for msg in unread:
                if msg.get("priority") == "high":
                    high_priority += 1

                email_insights = self._analyze_email(msg)
                insights.extend(email_insights)

            # Sort by priority
            priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
            insights.sort(key=lambda i: priority_order.get(i.priority, 99))

            # Generate natural language summary
            summary = self._generate_summary(total_unread, high_priority, insights)
            focus = self._suggest_focus(insights)

        except Exception as exc:
            logger.error("Proactive intelligence failed: %s", exc)
            summary = ""
            focus = ""

        briefing = DailyBriefing(
            timestamp=now,
            total_unread=total_unread,
            high_priority_count=high_priority,
            insights=insights[:20],  # Cap at 20 insights
            summary=summary,
            suggested_focus=focus,
        )

        self._last_briefing = briefing
        self._last_briefing_time = now
        return briefing

    def get_context_for_prompt(self) -> str:
        """Get the latest briefing formatted for system prompt injection."""
        briefing = self.generate_briefing()
        return briefing.to_prompt_context()

    def _get_unread_emails(self) -> list[dict]:
        """Get unread emails from the cache database."""
        # Access the email service's internal cache
        try:
            if hasattr(self._email, '_cache') and self._email._cache:
                conn = self._email._cache
                rows = conn.execute(
                    "SELECT id, subject, sender, snippet, priority, categories, date, labels "
                    "FROM emails WHERE is_read = 0 ORDER BY date DESC LIMIT 50"
                ).fetchall()
                return [
                    {"id": r[0], "subject": r[1], "sender": r[2], "snippet": r[3],
                     "priority": r[4], "categories": r[5], "date": r[6], "labels": r[7]}
                    for r in rows
                ]
            # Fallback: use email service search
            if hasattr(self._email, 'search'):
                results = self._email.search("is:unread", max_results=50)
                return [m.to_dict() if hasattr(m, 'to_dict') else m for m in results]
        except Exception as exc:
            logger.warning("Could not fetch unread emails: %s", exc)
        return []

    def _analyze_email(self, msg: dict) -> list[EmailInsight]:
        """Extract insights from a single email."""
        insights = []
        subject = msg.get("subject", "")
        sender = msg.get("sender", "")
        snippet = msg.get("snippet", "")
        priority = msg.get("priority", "medium")
        categories = msg.get("categories", "")

        # Sender name extraction
        sender_name = sender.split("<")[0].strip().strip('"') if "<" in sender else sender.split("@")[0]

        # Rule-based insight extraction
        subject_lower = subject.lower()
        snippet_lower = snippet.lower()
        combined = f"{subject_lower} {snippet_lower}"

        # Deadline detection
        deadline_keywords = ["deadline", "due by", "due date", "expires", "by end of", "before", "no later than"]
        if any(kw in combined for kw in deadline_keywords):
            insights.append(EmailInsight(
                category="deadline",
                priority="high" if priority == "high" else "medium",
                subject=f"Deadline mention: {subject[:60]}",
                detail=snippet[:200],
                source_email_id=msg.get("id", ""),
                source_subject=subject,
                source_sender=sender_name,
                suggested_action="Review and note the deadline in your calendar",
            ))

        # Action required detection
        action_keywords = ["action required", "please review", "please confirm", "your approval",
                          "sign off", "needs your", "waiting for your", "please respond",
                          "can you", "could you", "would you", "asap", "urgently"]
        if any(kw in combined for kw in action_keywords):
            urgency = "urgent" if any(kw in combined for kw in ["asap", "urgently", "immediately"]) else "high"
            insights.append(EmailInsight(
                category="action_required",
                priority=urgency,
                subject=f"Action needed: {subject[:60]}",
                detail=snippet[:200],
                source_email_id=msg.get("id", ""),
                source_subject=subject,
                source_sender=sender_name,
                suggested_action=f"Reply to {sender_name} or take requested action",
            ))

        # Financial / bills
        if categories and "bill" in categories:
            insights.append(EmailInsight(
                category="financial",
                priority="high",
                subject=f"Bill/Payment: {subject[:60]}",
                detail=snippet[:200],
                source_email_id=msg.get("id", ""),
                source_subject=subject,
                source_sender=sender_name,
                suggested_action="Review and pay if due",
            ))

        # Meeting/calendar related
        meeting_keywords = ["meeting", "calendar invite", "scheduled", "rsvp", "join us", "agenda"]
        if any(kw in combined for kw in meeting_keywords):
            insights.append(EmailInsight(
                category="important_update",
                priority="medium",
                subject=f"Meeting: {subject[:60]}",
                detail=snippet[:200],
                source_email_id=msg.get("id", ""),
                source_subject=subject,
                source_sender=sender_name,
                suggested_action="Check calendar and confirm attendance",
            ))

        # For high-priority emails with no rule-based insights,
        # try LLM analysis first, then fall back to generic insight
        if not insights and priority == "high":
            if self._inference_fn:
                llm_insight = self._llm_analyze(msg)
                if llm_insight:
                    insights.append(llm_insight)
            # Generic fallback if LLM unavailable or returned nothing
            if not insights:
                insights.append(EmailInsight(
                    category="important_update",
                    priority="high",
                    subject=subject[:60],
                    detail=snippet[:200],
                    source_email_id=msg.get("id", ""),
                    source_subject=subject,
                    source_sender=sender_name,
                ))

        return insights

    def _llm_analyze(self, msg: dict) -> Optional[EmailInsight]:
        """Use LLM to analyze ambiguous emails."""
        prompt = f"""Analyze this email and determine if it requires action.

Subject: {msg.get('subject', '')}
From: {msg.get('sender', '')}
Preview: {msg.get('snippet', '')[:300]}

Respond with a JSON object:
{{"needs_action": true/false, "category": "action_required"|"deadline"|"important_update"|"waiting_response"|"opportunity", "priority": "urgent"|"high"|"medium"|"low", "summary": "one sentence", "suggested_action": "what to do"}}"""

        try:
            raw = self._inference_fn(prompt=prompt, max_tokens=200, temperature=0.1)
            data = json.loads(raw)
            if data.get("needs_action"):
                sender = msg.get("sender", "")
                sender_name = sender.split("<")[0].strip().strip('"') if "<" in sender else sender.split("@")[0]
                return EmailInsight(
                    category=data.get("category", "important_update"),
                    priority=data.get("priority", "medium"),
                    subject=data.get("summary", msg.get("subject", ""))[:60],
                    detail=msg.get("snippet", "")[:200],
                    source_email_id=msg.get("id", ""),
                    source_subject=msg.get("subject", ""),
                    source_sender=sender_name,
                    suggested_action=data.get("suggested_action", ""),
                )
        except Exception as exc:
            logger.debug("LLM email analysis failed: %s", exc)
        return None

    def _generate_summary(self, total_unread: int, high_priority: int, insights: list[EmailInsight]) -> str:
        """Generate a one-line natural language summary."""
        if total_unread == 0:
            return "Inbox is clear — no unread emails."

        parts = []
        urgent = [i for i in insights if i.priority == "urgent"]
        actions = [i for i in insights if i.category == "action_required"]
        deadlines = [i for i in insights if i.category == "deadline"]

        if urgent:
            parts.append(f"{len(urgent)} urgent item{'s' if len(urgent) > 1 else ''} need immediate attention")
        if actions:
            parts.append(f"{len(actions)} email{'s' if len(actions) > 1 else ''} need your response")
        if deadlines:
            parts.append(f"{len(deadlines)} deadline{'s' if len(deadlines) > 1 else ''} mentioned")

        if parts:
            return f"{total_unread} unread ({high_priority} high-priority): " + "; ".join(parts) + "."
        return f"{total_unread} unread emails, {high_priority} high-priority."

    def _suggest_focus(self, insights: list[EmailInsight]) -> str:
        """Suggest what to focus on based on insights."""
        urgent = [i for i in insights if i.priority == "urgent"]
        if urgent:
            return f"Handle urgent: {urgent[0].subject}"

        actions = [i for i in insights if i.category == "action_required"]
        if actions:
            return f"Respond to: {actions[0].source_sender} — {actions[0].subject}"

        deadlines = [i for i in insights if i.category == "deadline"]
        if deadlines:
            return f"Check deadline: {deadlines[0].subject}"

        return "Review your inbox when you have a moment"
