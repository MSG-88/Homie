"""Tests for /briefing slash command."""
import pytest
from unittest.mock import MagicMock
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.briefing import register


def test_briefing_no_sources():
    router = SlashCommandRouter()
    register(router, {})

    cfg = MagicMock()
    cfg.location = None
    cfg.user_name = "Master"

    vault = MagicMock()
    vault.get_credential.return_value = None

    result = router.dispatch("/briefing", **{"config": cfg, "vault": vault, "_router": router})
    assert "Master" in result
    assert "No data sources" in result


def test_briefing_with_greeting():
    router = SlashCommandRouter()
    register(router, {})

    cfg = MagicMock()
    cfg.location = None
    cfg.user_name = "Tester"

    result = router.dispatch("/briefing", **{"config": cfg, "vault": MagicMock(get_credential=MagicMock(return_value=None)), "_router": router})
    assert "Tester" in result
    # Should have either "morning" or "day" greeting
    assert "Good" in result
