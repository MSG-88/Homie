"""Smart greeting -- Jarvis-style session opening with full context.

Generates an intelligent greeting that includes email briefing,
active project status, and proactive suggestions based on the
user's patterns and current context.
"""
from __future__ import annotations

import logging
from typing import Optional

from homie_core.context.awareness import ContextualAwareness, SessionContext

logger = logging.getLogger(__name__)


class SmartGreeting:
    """Generates context-aware session greetings."""

    def __init__(self, awareness: ContextualAwareness):
        self._awareness = awareness

    def generate(self) -> str:
        """Generate a full Jarvis-style greeting with context.

        Returns a rich greeting string that Homie displays at session start.
        """
        ctx = self._awareness.refresh(force=True)

        sections = []

        # 1. Time-aware greeting
        sections.append(self._greeting_line(ctx))

        # 2. Email status (if connected)
        if ctx.email_briefing:
            sections.append(ctx.email_briefing)
        elif ctx.unread_count > 0:
            sections.append(f"You have {ctx.unread_count} unread emails.")

        # 3. Active project
        if ctx.active_project:
            project_line = f"You're in the **{ctx.active_project}** project."
            if ctx.recent_git_activity:
                project_line += f" Recent: {ctx.recent_git_activity.split(';')[0]}"
            sections.append(project_line)

        # 4. Connected services summary
        if ctx.connected_services:
            services = ", ".join(s.title() for s in ctx.connected_services)
            sections.append(f"Connected to: {services}")

        # 5. Proactive suggestion
        sections.append(self._suggest(ctx))

        return "\n\n".join(s for s in sections if s)

    def _greeting_line(self, ctx: SessionContext) -> str:
        """Generate the opening greeting line."""
        greetings = {
            "late_night": f"Still up, {ctx.user_name}? I'm here if you need me.",
            "morning": f"Good morning, {ctx.user_name}! Ready to start the day.",
            "afternoon": f"Good afternoon, {ctx.user_name}. Let's get things done.",
            "evening": f"Good evening, {ctx.user_name}. Winding down for the day?",
            "night": f"Good evening, {ctx.user_name}. How can I help tonight?",
        }
        return greetings.get(ctx.time_of_day, f"Hello, {ctx.user_name}!")

    def _suggest(self, ctx: SessionContext) -> str:
        """Generate a proactive suggestion based on context."""
        suggestions = []

        if ctx.urgent_emails > 0:
            suggestions.append(f"handle the {ctx.urgent_emails} urgent email{'s' if ctx.urgent_emails > 1 else ''}")

        if ctx.active_project and ctx.recent_git_activity:
            suggestions.append(f"continue work on {ctx.active_project}")

        if not suggestions:
            suggestions.append("let me know what you'd like to focus on")

        return "I'd suggest you " + " or ".join(suggestions) + "."
