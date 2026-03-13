"""Tests for individual slash command handlers."""
import pytest
from unittest.mock import MagicMock
from homie_app.console.router import SlashCommandRouter, SlashCommand
from homie_app.console.commands.help import register as register_help
from homie_app.console.commands.memory import register as register_memory


def test_help_lists_all_commands():
    router = SlashCommandRouter()
    ctx = {"_router": router}
    register_help(router, ctx)
    router.register(SlashCommand(name="test", description="A test"))
    result = router.dispatch("/help", **ctx)
    assert "test" in result
    assert "help" in result


def test_help_specific_command():
    router = SlashCommandRouter()
    ctx = {"_router": router}
    register_help(router, ctx)
    router.register(SlashCommand(name="connect", description="Connect a provider"))
    result = router.dispatch("/help connect", **ctx)
    assert "connect" in result.lower()


def test_status_shows_memory_info():
    wm = MagicMock()
    wm.get_conversation.return_value = ["msg1", "msg2"]
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "test", "confidence": 0.9}]
    cfg = MagicMock()
    cfg.user_name = "Tester"

    router = SlashCommandRouter()
    ctx = {"wm": wm, "sm": sm, "em": None, "config": cfg, "_router": router}
    register_memory(router, ctx)
    result = router.dispatch("/status", **ctx)
    assert "2 messages" in result
    assert "Tester" in result


def test_remember_stores_fact():
    sm = MagicMock()
    router = SlashCommandRouter()
    ctx = {"wm": MagicMock(), "sm": sm, "config": MagicMock(), "_router": router}
    register_memory(router, ctx)
    result = router.dispatch("/remember I like coffee", **ctx)
    sm.learn.assert_called_once_with("I like coffee", confidence=0.9, tags=["user_explicit"])
    assert "remember" in result.lower() or "coffee" in result.lower()


def test_clear_clears_working_memory():
    wm = MagicMock()
    router = SlashCommandRouter()
    ctx = {"wm": wm, "sm": None, "config": MagicMock(), "_router": router}
    register_memory(router, ctx)
    result = router.dispatch("/clear", **ctx)
    wm.clear.assert_called_once()
    assert "clear" in result.lower() or "fresh" in result.lower()


def test_forget_topic():
    sm = MagicMock()
    router = SlashCommandRouter()
    ctx = {"sm": sm, "_router": router}
    register_memory(router, ctx)
    result = router.dispatch("/forget work", **ctx)
    sm.forget_topic.assert_called_once_with("work")
    assert "work" in result.lower()


def test_facts_no_memory():
    router = SlashCommandRouter()
    ctx = {"sm": None, "_router": router}
    register_memory(router, ctx)
    result = router.dispatch("/facts", **ctx)
    assert "not available" in result.lower()
