"""Tests for /connect, /disconnect, /connections slash commands."""
import pytest
from unittest.mock import MagicMock, patch
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.connect import register


def test_connections_lists_providers():
    router = SlashCommandRouter()
    vault = MagicMock()
    conn1 = MagicMock(provider="gmail", connected=True, display_label="user@gmail.com")
    conn2 = MagicMock(provider="linkedin", connected=False, display_label=None)
    vault.get_all_connections.return_value = [conn1, conn2]
    ctx = {"config": MagicMock(), "vault": vault, "_router": router}
    register(router, ctx)

    result = router.dispatch("/connections", **ctx)
    assert "gmail" in result
    assert "linkedin" in result


def test_connect_no_provider_shows_usage():
    router = SlashCommandRouter()
    ctx = {"config": MagicMock(), "vault": MagicMock(), "_router": router}
    register(router, ctx)
    result = router.dispatch("/connect", **ctx)
    assert "provider" in result.lower() or "usage" in result.lower()


def test_connect_unknown_provider():
    router = SlashCommandRouter()
    ctx = {"config": MagicMock(), "vault": MagicMock(), "_router": router}
    register(router, ctx)
    result = router.dispatch("/connect nonexistent", **ctx)
    assert "Unknown provider" in result


def test_disconnect_provider():
    router = SlashCommandRouter()
    vault = MagicMock()
    ctx = {"config": MagicMock(), "vault": vault, "_router": router}
    register(router, ctx)
    result = router.dispatch("/disconnect gmail", **ctx)
    vault.set_connection_status.assert_called_once_with("gmail", connected=False)
    assert "gmail" in result.lower()


def test_disconnect_no_provider():
    router = SlashCommandRouter()
    ctx = {"config": MagicMock(), "vault": MagicMock(), "_router": router}
    register(router, ctx)
    result = router.dispatch("/disconnect", **ctx)
    assert "usage" in result.lower()
