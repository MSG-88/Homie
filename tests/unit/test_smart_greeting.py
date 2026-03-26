"""Tests for smart greeting generation."""
import pytest
from unittest.mock import MagicMock
from homie_core.brain.smart_greeting import SmartGreeting
from homie_core.context.awareness import ContextualAwareness, SessionContext


class TestSmartGreeting:
    def _mock_awareness(self, **overrides):
        defaults = dict(
            user_name="Muthu",
            time_of_day="morning",
            current_time="9:30 AM",
            day_of_week="Monday",
        )
        defaults.update(overrides)
        ctx = SessionContext(**defaults)
        awareness = MagicMock(spec=ContextualAwareness)
        awareness.refresh.return_value = ctx
        return awareness

    def test_basic_greeting(self):
        awareness = self._mock_awareness()
        sg = SmartGreeting(awareness)
        greeting = sg.generate()
        assert "Good morning" in greeting
        assert "Muthu" in greeting

    def test_includes_email_briefing(self):
        awareness = self._mock_awareness(
            email_briefing="3 unread, 1 urgent", unread_count=3, urgent_emails=1,
        )
        sg = SmartGreeting(awareness)
        greeting = sg.generate()
        assert "3 unread" in greeting

    def test_includes_project(self):
        awareness = self._mock_awareness(
            active_project="Homie", recent_git_activity="feat: add smart greeting",
        )
        sg = SmartGreeting(awareness)
        greeting = sg.generate()
        assert "Homie" in greeting

    def test_includes_services(self):
        awareness = self._mock_awareness(connected_services=["gmail", "linkedin"])
        sg = SmartGreeting(awareness)
        greeting = sg.generate()
        assert "Gmail" in greeting
        assert "Linkedin" in greeting

    def test_suggests_urgent_emails(self):
        awareness = self._mock_awareness(urgent_emails=2)
        sg = SmartGreeting(awareness)
        greeting = sg.generate()
        assert "urgent" in greeting.lower()

    def test_suggests_project_work(self):
        awareness = self._mock_awareness(
            active_project="Homie", recent_git_activity="fix: something",
        )
        sg = SmartGreeting(awareness)
        greeting = sg.generate()
        assert "Homie" in greeting

    def test_evening_greeting(self):
        awareness = self._mock_awareness(time_of_day="evening")
        sg = SmartGreeting(awareness)
        greeting = sg.generate()
        assert "Good evening" in greeting

    def test_late_night_greeting(self):
        awareness = self._mock_awareness(time_of_day="late_night")
        sg = SmartGreeting(awareness)
        greeting = sg.generate()
        assert "Still up" in greeting
