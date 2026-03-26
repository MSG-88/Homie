"""Tests for contextual awareness engine."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from homie_core.context.awareness import ContextualAwareness, SessionContext


class TestSessionContext:
    def test_default_values(self):
        ctx = SessionContext()
        assert ctx.user_name == "Master"
        assert ctx.unread_count == 0

    def test_to_system_context(self):
        ctx = SessionContext(
            user_name="Muthu", current_time="10:30 AM",
            day_of_week="Monday", time_of_day="morning",
            email_briefing="3 unread", active_project="Homie",
            connected_services=["gmail", "linkedin"],
        )
        text = ctx.to_system_context()
        assert "10:30 AM" in text
        assert "Homie" in text
        assert "gmail" in text

    def test_to_greeting_morning(self):
        ctx = SessionContext(
            user_name="Muthu", time_of_day="morning",
            unread_count=5, urgent_emails=1,
            active_project="Homie",
        )
        greeting = ctx.to_greeting()
        assert "Good morning" in greeting
        assert "Muthu" in greeting
        assert "5 unread" in greeting
        assert "1 marked urgent" in greeting
        assert "Homie" in greeting

    def test_to_greeting_empty_inbox(self):
        ctx = SessionContext(user_name="Muthu", time_of_day="evening")
        greeting = ctx.to_greeting()
        assert "Good evening" in greeting
        assert "unread" not in greeting


class TestContextualAwareness:
    def test_refresh_basic(self):
        engine = ContextualAwareness(user_name="Muthu")
        ctx = engine.refresh(force=True)
        assert ctx.user_name == "Muthu"
        assert ctx.time_of_day  # Should be detected from system clock

    def test_email_integration(self):
        mock_intel = MagicMock()
        mock_briefing = MagicMock()
        mock_briefing.to_prompt_context.return_value = "3 unread emails"
        mock_briefing.total_unread = 3
        mock_briefing.insights = []
        mock_intel.generate_briefing.return_value = mock_briefing

        engine = ContextualAwareness(user_name="Muthu", email_intelligence=mock_intel)
        ctx = engine.refresh(force=True)
        assert ctx.unread_count == 3
        assert "3 unread" in ctx.email_briefing

    def test_vault_services(self):
        mock_vault = MagicMock()
        conn1 = MagicMock(); conn1.provider = "gmail"; conn1.connected = True
        conn2 = MagicMock(); conn2.provider = "linkedin"; conn2.connected = True
        conn3 = MagicMock(); conn3.provider = "twitter"; conn3.connected = False
        mock_vault.get_all_connections.return_value = [conn1, conn2, conn3]

        engine = ContextualAwareness(user_name="Muthu", vault=mock_vault)
        ctx = engine.refresh(force=True)
        assert "gmail" in ctx.connected_services
        assert "linkedin" in ctx.connected_services
        assert "twitter" not in ctx.connected_services

    def test_get_greeting(self):
        engine = ContextualAwareness(user_name="Muthu")
        greeting = engine.get_greeting()
        assert "Muthu" in greeting
        assert "focus on" in greeting.lower()

    def test_caching(self):
        engine = ContextualAwareness(user_name="Muthu")
        ctx1 = engine.refresh(force=True)
        ctx2 = engine.refresh()  # cached
        assert ctx1 is ctx2

    def test_get_system_prompt(self):
        engine = ContextualAwareness(user_name="Muthu")
        with patch("homie_app.prompts.system.build_system_prompt", return_value="test prompt"):
            prompt = engine.get_system_prompt()
            assert prompt == "test prompt"
