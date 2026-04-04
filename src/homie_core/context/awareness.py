"""Contextual Awareness Engine — Homie's unified context aggregator.

Combines email intelligence, calendar, projects, preferences, and
behavioral patterns into a single context object that powers
intelligent, proactive responses.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Everything Homie knows at this moment."""
    user_name: str = "Master"
    time_of_day: str = ""  # morning/afternoon/evening/night
    day_of_week: str = ""
    current_time: str = ""

    # Email context
    email_briefing: str = ""
    unread_count: int = 0
    urgent_emails: int = 0

    # Knowledge
    known_facts: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)

    # Preferences
    verbosity: str = "balanced"   # concise/balanced/detailed
    formality: str = "casual"     # casual/neutral/formal
    technical_depth: str = "moderate"  # simple/moderate/expert

    # Activity
    active_project: str = ""
    recent_git_activity: str = ""

    # Connected services
    connected_services: list[str] = field(default_factory=list)

    # Behavioral
    typical_activity: str = ""  # What user usually does at this time

    def to_system_context(self) -> str:
        """Build a rich context block for the system prompt."""
        lines = []

        # Time awareness
        lines.append(f"Current time: {self.current_time} ({self.day_of_week} {self.time_of_day})")

        # Email briefing
        if self.email_briefing:
            lines.append(self.email_briefing)

        # Active project
        if self.active_project:
            lines.append(f"Currently working on: {self.active_project}")
        if self.recent_git_activity:
            lines.append(f"Recent git: {self.recent_git_activity}")

        # Connected services
        if self.connected_services:
            lines.append(f"Connected: {', '.join(self.connected_services)}")

        # Behavioral pattern
        if self.typical_activity:
            lines.append(f"User typically: {self.typical_activity}")

        return "\n".join(lines)

    def to_greeting(self) -> str:
        """Generate a Jarvis-style proactive greeting."""
        greetings = {
            "morning": "Good morning",
            "afternoon": "Good afternoon",
            "evening": "Good evening",
            "night": "Good evening",
            "late_night": "Burning the midnight oil",
        }
        greeting = greetings.get(self.time_of_day, "Hello")

        parts = [f"{greeting}, {self.user_name}."]

        # Email summary
        if self.unread_count > 0:
            if self.urgent_emails > 0:
                parts.append(f"You have {self.unread_count} unread emails, {self.urgent_emails} marked urgent.")
            else:
                parts.append(f"You have {self.unread_count} unread emails.")

        # Active project
        if self.active_project:
            parts.append(f"You were last working on {self.active_project}.")

        # Behavioral suggestion
        if self.typical_activity:
            parts.append(f"Around this time you usually {self.typical_activity}.")

        parts.append("What would you like to focus on?")

        return " ".join(parts)


class ContextualAwareness:
    """Aggregates all context sources into a unified SessionContext.

    Call refresh() to update, then access .context for the latest state.
    """

    def __init__(
        self,
        user_name: str = "Master",
        email_intelligence=None,    # ProactiveEmailIntelligence instance
        semantic_memory=None,        # SemanticMemory instance
        preference_engine=None,      # PreferenceEngine instance
        vault=None,                  # SecureVault for connection status
    ):
        self._user_name = user_name
        self._email_intel = email_intelligence
        self._semantic = semantic_memory
        self._prefs = preference_engine
        self._vault = vault
        self._context = SessionContext(user_name=user_name)
        self._last_refresh: float = 0
        self._refresh_interval = 300  # 5 min

    @property
    def context(self) -> SessionContext:
        return self._context

    def refresh(self, force: bool = False) -> SessionContext:
        """Refresh all context sources."""
        now = time.time()
        if not force and (now - self._last_refresh) < self._refresh_interval:
            return self._context

        ctx = SessionContext(user_name=self._user_name)

        # Time context
        dt = datetime.now()
        ctx.current_time = dt.strftime("%I:%M %p")
        ctx.day_of_week = dt.strftime("%A")
        hour = dt.hour
        if hour < 6:
            ctx.time_of_day = "late_night"
        elif hour < 12:
            ctx.time_of_day = "morning"
        elif hour < 17:
            ctx.time_of_day = "afternoon"
        elif hour < 21:
            ctx.time_of_day = "evening"
        else:
            ctx.time_of_day = "night"

        # Email intelligence
        if self._email_intel:
            try:
                briefing = self._email_intel.generate_briefing()
                ctx.email_briefing = briefing.to_prompt_context()
                ctx.unread_count = briefing.total_unread
                ctx.urgent_emails = len([i for i in briefing.insights if i.priority == "urgent"])
            except Exception as exc:
                logger.debug("Email context failed: %s", exc)

        # Known facts from semantic memory
        if self._semantic:
            try:
                facts = self._semantic.get_facts(min_confidence=0.5)
                ctx.known_facts = [f["fact"] for f in facts[:10]]
            except Exception as exc:
                logger.debug("Semantic memory failed: %s", exc)

        # Preferences
        if self._prefs:
            try:
                if hasattr(self._prefs, 'get_preferences'):
                    prefs = self._prefs.get_preferences()
                    ctx.verbosity = prefs.get("verbosity", "balanced")
                    ctx.formality = prefs.get("formality", "casual")
                    ctx.technical_depth = prefs.get("technical_depth", "moderate")
            except Exception as exc:
                logger.debug("Preference engine failed: %s", exc)

        # Connected services from vault
        if self._vault:
            try:
                conns = self._vault.get_all_connections()
                ctx.connected_services = [c.provider for c in conns if c.connected]
            except Exception as exc:
                logger.debug("Vault connection check failed: %s", exc)

        # Active project detection (from CWD or git)
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                import os
                ctx.active_project = os.path.basename(result.stdout.strip())

            # Recent git activity
            result = subprocess.run(
                ["git", "log", "--oneline", "-3", "--format=%s"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                commits = result.stdout.strip().split("\n")[:3]
                ctx.recent_git_activity = "; ".join(commits)
        except Exception as e:
            logger.debug("Git context refresh failed: %s", e)

        self._context = ctx
        self._last_refresh = now
        return ctx

    def get_system_prompt(self) -> str:
        """Build a complete system prompt with all context injected."""
        ctx = self.refresh()
        from homie_app.prompts.system import build_system_prompt
        # Use full system context (includes email, project, services, git)
        full_context = ctx.to_system_context()
        return build_system_prompt(
            user_name=ctx.user_name,
            time_of_day=ctx.time_of_day,
            known_facts=ctx.known_facts,
            email_context=full_context,
        )

    def get_greeting(self) -> str:
        """Generate a proactive greeting with full context."""
        ctx = self.refresh(force=True)
        return ctx.to_greeting()
