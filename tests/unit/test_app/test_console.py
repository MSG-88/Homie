"""Tests for Console main loop — input routing, init detection, quit."""
import pytest
from unittest.mock import MagicMock, patch
from homie_app.console.console import Console
from homie_app.console.router import SlashCommand


def test_slash_command_routes_to_router(tmp_path):
    """Slash commands dispatch through the router, not the brain."""
    config = MagicMock()
    config.storage.path = str(tmp_path)
    config.user_name = "Tester"

    console = Console(config=config, skip_init=True)
    console._router.register(
        SlashCommand(name="test", description="test", handler_fn=lambda args, **ctx: "routed")
    )

    with patch("builtins.input", side_effect=["/test", "quit"]):
        with patch.object(console, "_print") as mock_print:
            console.run()
            printed = [str(c) for c in mock_print.call_args_list]
            assert any("routed" in p for p in printed)


def test_quit_exits_loop(tmp_path):
    """Typing 'quit' exits the console loop."""
    config = MagicMock()
    config.storage.path = str(tmp_path)
    config.user_name = "Tester"

    console = Console(config=config, skip_init=True)
    with patch("builtins.input", side_effect=["quit"]):
        console.run()  # Should not hang


def test_empty_input_skipped(tmp_path):
    """Empty input is ignored, loop continues."""
    config = MagicMock()
    config.storage.path = str(tmp_path)
    config.user_name = "Tester"

    console = Console(config=config, skip_init=True)
    with patch("builtins.input", side_effect=["", "quit"]):
        console.run()  # Should not error


def test_slash_quit_exits_loop(tmp_path):
    """Typing '/quit' exits the console loop."""
    config = MagicMock()
    config.storage.path = str(tmp_path)
    config.user_name = "Tester"

    console = Console(config=config, skip_init=True)
    with patch("builtins.input", side_effect=["/quit"]):
        console.run()  # Should not hang


def test_no_brain_shows_message(tmp_path):
    """Chat input without brain shows error message."""
    config = MagicMock()
    config.storage.path = str(tmp_path)
    config.user_name = "Tester"

    console = Console(config=config, skip_init=True)
    with patch("builtins.input", side_effect=["hello world", "quit"]):
        with patch.object(console, "_print") as mock_print:
            console.run()
            printed = [str(c) for c in mock_print.call_args_list]
            assert any("No model loaded" in p for p in printed)
