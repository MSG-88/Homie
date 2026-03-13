"""Integration tests for the unified console."""
import pytest
from unittest.mock import MagicMock, patch


def test_full_console_slash_command_flow():
    """Test: launch console, run slash commands, quit."""
    from homie_app.console import Console
    from homie_app.console.router import SlashCommand

    cfg = MagicMock()
    cfg.user_name = "Tester"
    cfg.storage.path = "/tmp/.homie"
    cfg.location = None

    console = Console(config=cfg, skip_init=True)
    # Register a test command
    console._router.register(SlashCommand(
        name="test", description="Test", handler_fn=lambda args, **ctx: f"echo:{args}"
    ))

    inputs = ["/test hello", "/", "quit"]
    with patch("builtins.input", side_effect=inputs):
        with patch.dict("sys.modules", {"prompt_toolkit": None}):
            with patch.object(console, "_print") as mock_print:
                console.run()

    printed = " ".join(str(c) for c in mock_print.call_args_list)
    assert "echo:hello" in printed
    assert "Available" in printed or "Commands" in printed


def test_console_first_run_triggers_wizard():
    """Test: no model configured triggers wizard."""
    from homie_app.console import Console

    cfg = MagicMock()
    cfg.user_name = ""
    cfg.llm.model_path = ""
    cfg.storage.path = "/tmp/.homie"

    with patch.object(Console, "_run_wizard") as mock_wizard:
        with patch.object(Console, "run"):
            console = Console(config=cfg)
            # Wizard should have been called during bootstrap
            mock_wizard.assert_called_once()


def test_console_all_commands_registered():
    """Test: all expected command modules are registered."""
    from homie_app.console import Console

    cfg = MagicMock()
    cfg.user_name = "Tester"

    console = Console(config=cfg, skip_init=True)
    # Register all commands manually
    from homie_app.console.commands import register_all_commands
    ctx = {"config": cfg, "_router": console._router, "vault": None}
    register_all_commands(console._router, ctx)

    cmd_names = [c.name for c in console._router.list_commands()]
    expected = [
        "help", "quit", "connect", "email", "settings",
        "location", "weather", "news", "briefing",
        "model", "plugins", "daemon", "backup",
    ]
    for name in expected:
        assert name in cmd_names, f"Missing command: /{name}"


def test_console_autocomplete_integration():
    """Test: autocomplete returns commands for '/' prefix."""
    from homie_app.console.autocomplete import HomieCompleter
    from homie_app.console.router import SlashCommandRouter, SlashCommand
    from prompt_toolkit.document import Document

    router = SlashCommandRouter()
    router.register(SlashCommand(name="help", description="Help"))
    router.register(SlashCommand(name="weather", description="Weather"))
    router.register(SlashCommand(name="news", description="News"))

    completer = HomieCompleter(router)
    doc = Document("/w", cursor_position=2)
    completions = list(completer.get_completions(doc, None))
    texts = [c.text for c in completions]
    assert "weather" in texts
    assert "news" not in texts
