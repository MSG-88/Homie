from unittest.mock import MagicMock, patch

from homie_core.config import HomieConfig
from homie_app.cli import create_parser, main, _handle_meta_command


def test_parser_creation():
    parser = create_parser()
    assert parser is not None


def test_parse_model_list():
    parser = create_parser()
    args = parser.parse_args(["model", "list"])
    assert args.command == "model"
    assert args.model_command == "list"


def test_parse_plugin_enable():
    parser = create_parser()
    args = parser.parse_args(["plugin", "enable", "email"])
    assert args.command == "plugin"
    assert args.plugin_command == "enable"
    assert args.name == "email"


def test_parse_backup():
    parser = create_parser()
    args = parser.parse_args(["backup", "--to", "/tmp/backup"])
    assert args.command == "backup"
    assert args.backup_path == "/tmp/backup"


def test_parse_chat():
    parser = create_parser()
    args = parser.parse_args(["chat"])
    assert args.command == "chat"


def test_no_command_prints_help(capsys):
    main([])
    captured = capsys.readouterr()
    assert "homie" in captured.out.lower() or "usage" in captured.out.lower()


# -----------------------------------------------------------------------
# Meta-command tests
# -----------------------------------------------------------------------

class TestMetaCommands:
    def setup_method(self):
        from homie_core.memory.working import WorkingMemory
        self.wm = WorkingMemory()
        self.sm = MagicMock()
        self.em = MagicMock()
        self.brain = MagicMock()
        self.brain._cognitive._learning.get_session_stats.return_value = {
            "interactions": 5, "facts_learned": 2, "facts": ["User likes Python", "User is a dev"],
        }
        self.cfg = MagicMock()
        self.cfg.user_name = "TestUser"

    def test_help_command(self):
        result = _handle_meta_command("/help", self.brain, self.wm, self.sm, self.em, self.cfg)
        assert result is not None
        assert "/status" in result
        assert "/facts" in result

    def test_status_command(self):
        self.sm.get_facts.return_value = [{"fact": "test", "confidence": 0.8}]
        result = _handle_meta_command("/status", self.brain, self.wm, self.sm, self.em, self.cfg)
        assert "Homie Status" in result
        assert "Facts stored: 1" in result

    def test_remember_command(self):
        self.sm.learn.return_value = 1
        result = _handle_meta_command("/remember I prefer dark mode", self.brain, self.wm, self.sm, self.em, self.cfg)
        assert "remember" in result.lower()
        self.sm.learn.assert_called_once_with("I prefer dark mode", confidence=0.9, tags=["user_explicit"])

    def test_forget_command(self):
        result = _handle_meta_command("/forget work", self.brain, self.wm, self.sm, self.em, self.cfg)
        assert "Forgotten" in result
        self.sm.forget_topic.assert_called_once_with("work")

    def test_facts_command(self):
        self.sm.get_facts.return_value = [
            {"fact": "User likes Python", "confidence": 0.9},
            {"fact": "User prefers dark mode", "confidence": 0.8},
        ]
        result = _handle_meta_command("/facts", self.brain, self.wm, self.sm, self.em, self.cfg)
        assert "Python" in result
        assert "dark mode" in result

    def test_learn_command(self):
        result = _handle_meta_command("/learn", self.brain, self.wm, self.sm, self.em, self.cfg)
        assert "Session Learning Stats" in result
        assert "Interactions: 5" in result

    def test_non_command_returns_none(self):
        result = _handle_meta_command("hello", self.brain, self.wm, self.sm, self.em, self.cfg)
        assert result is None


# -----------------------------------------------------------------------
# Dynamic system prompt tests
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


# -----------------------------------------------------------------------
# Stop and Status command tests
# -----------------------------------------------------------------------

class TestStopCommand:
    def test_parser_has_stop(self):
        parser = create_parser()
        args = parser.parse_args(["stop"])
        assert args.command == "stop"


class TestStatusCommand:
    def test_parser_has_status(self):
        parser = create_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"
