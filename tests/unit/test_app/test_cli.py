"""Tests for the simplified CLI entry point."""
from unittest.mock import MagicMock, patch

from homie_app.cli import create_parser


def test_parser_creation():
    parser = create_parser()
    assert parser is not None


def test_parser_no_args():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_parser_start():
    parser = create_parser()
    args = parser.parse_args(["start"])
    assert args.command == "start"


def test_parser_config_flag():
    parser = create_parser()
    args = parser.parse_args(["--config", "/path/to/config.yaml"])
    assert args.config == "/path/to/config.yaml"


def test_parser_no_voice_flag():
    parser = create_parser()
    args = parser.parse_args(["--no-voice"])
    assert args.no_voice is True


def test_parser_no_tray_flag():
    parser = create_parser()
    args = parser.parse_args(["--no-tray"])
    assert args.no_tray is True


def test_start_with_config():
    parser = create_parser()
    args = parser.parse_args(["start", "--config", "/tmp/config.yaml"])
    assert args.command == "start"
    assert args.start_config == "/tmp/config.yaml"


# -----------------------------------------------------------------------
# Dynamic system prompt tests (unchanged — these test prompts, not CLI)
# -----------------------------------------------------------------------

class TestDynamicSystemPrompt:
    def test_build_system_prompt_basic(self):
        from homie_app.prompts.system import build_system_prompt
        prompt = build_system_prompt(user_name="Alice")
        assert "Alice" in prompt
        assert "Homie" in prompt
        assert "privacy" in prompt.lower()

    def test_build_system_prompt_with_facts(self):
        from homie_app.prompts.system import build_system_prompt
        prompt = build_system_prompt(
            user_name="Bob",
            known_facts=["Bob likes Python", "Bob works at Acme Corp"],
        )
        assert "Bob likes Python" in prompt
        assert "Acme Corp" in prompt

    def test_build_system_prompt_time_of_day(self):
        from homie_app.prompts.system import build_system_prompt
        prompt = build_system_prompt(user_name="Test", time_of_day="morning")
        assert "morning" in prompt.lower() or "upbeat" in prompt.lower()

    def test_system_prompt_fallback(self):
        from homie_app.prompts.system import SYSTEM_PROMPT
        assert "Homie" in SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 200
