"""Tests for /settings slash command."""
from unittest.mock import MagicMock, patch
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.settings import register


def test_settings_registered():
    router = SlashCommandRouter()
    register(router, {})
    cmd = router._commands.get("settings")
    assert cmd is not None
    assert "settings" in cmd.name


def test_settings_dispatches():
    router = SlashCommandRouter()
    register(router, {})
    cfg = MagicMock()
    cfg.user_name = "Tester"
    # Settings menu is interactive (uses input()), so mock with "Back" option
    with patch("builtins.input", return_value="10"):
        result = router.dispatch("/settings", config=cfg, _router=router)
    assert result is not None
