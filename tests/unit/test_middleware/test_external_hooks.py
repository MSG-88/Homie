from __future__ import annotations

import subprocess
import pytest
from unittest.mock import patch, MagicMock, call

from homie_core.middleware.external_hooks import ExternalHooksMiddleware


# ---------------------------------------------------------------------------
# No hooks configured — passthrough, no errors
# ---------------------------------------------------------------------------

def test_no_hooks_before_turn_passthrough():
    mw = ExternalHooksMiddleware()
    result = mw.before_turn("hello", {})
    assert result == "hello"


def test_no_hooks_after_turn_passthrough():
    mw = ExternalHooksMiddleware()
    result = mw.after_turn("response text", {})
    assert result == "response text"


def test_no_hooks_wrap_tool_call_passthrough():
    mw = ExternalHooksMiddleware()
    args = {"key": "value"}
    result = mw.wrap_tool_call("my_tool", args)
    assert result == args


def test_no_hooks_wrap_tool_result_passthrough():
    mw = ExternalHooksMiddleware()
    result = mw.wrap_tool_result("my_tool", "some result")
    assert result == "some result"


def test_empty_hooks_dict_no_errors():
    mw = ExternalHooksMiddleware(hooks={})
    mw.before_turn("msg", {})
    mw.after_turn("resp", {})
    mw.wrap_tool_call("tool", {})
    mw.wrap_tool_result("tool", "res")


# ---------------------------------------------------------------------------
# before_turn fires configured command
# ---------------------------------------------------------------------------

def test_before_turn_fires_command(tmp_path):
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(hooks={"before_turn": ["echo before"]})
        mw.before_turn("some message", {})
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == "echo before"


def test_before_turn_returns_message_unchanged():
    with patch("subprocess.run"):
        mw = ExternalHooksMiddleware(hooks={"before_turn": ["echo hi"]})
        result = mw.before_turn("original message", {})
        assert result == "original message"


def test_before_turn_uses_shell_true():
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(hooks={"before_turn": ["echo test"]})
        mw.before_turn("msg", {})
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is True


# ---------------------------------------------------------------------------
# after_turn fires configured command
# ---------------------------------------------------------------------------

def test_after_turn_fires_command():
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(hooks={"after_turn": ["echo after"]})
        mw.after_turn("some response", {})
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == "echo after"


def test_after_turn_returns_response_unchanged():
    with patch("subprocess.run"):
        mw = ExternalHooksMiddleware(hooks={"after_turn": ["echo done"]})
        result = mw.after_turn("original response", {})
        assert result == "original response"


# ---------------------------------------------------------------------------
# on_tool_call and on_tool_result events
# ---------------------------------------------------------------------------

def test_wrap_tool_call_fires_on_tool_call_event():
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(hooks={"on_tool_call": ["echo tool_called"]})
        mw.wrap_tool_call("read_file", {"path": "/tmp/file"})
        mock_run.assert_called_once()


def test_wrap_tool_call_returns_args_unchanged():
    with patch("subprocess.run"):
        mw = ExternalHooksMiddleware(hooks={"on_tool_call": ["echo x"]})
        args = {"path": "/tmp/file"}
        result = mw.wrap_tool_call("read_file", args)
        assert result == args


def test_wrap_tool_result_fires_on_tool_result_event():
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(hooks={"on_tool_result": ["echo result_fired"]})
        mw.wrap_tool_result("read_file", "file contents")
        mock_run.assert_called_once()


def test_wrap_tool_result_returns_result_unchanged():
    with patch("subprocess.run"):
        mw = ExternalHooksMiddleware(hooks={"on_tool_result": ["echo x"]})
        result = mw.wrap_tool_result("read_file", "the content")
        assert result == "the content"


# ---------------------------------------------------------------------------
# Timeout doesn't crash (5s timeout)
# ---------------------------------------------------------------------------

def test_timeout_does_not_crash():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
        mw = ExternalHooksMiddleware(hooks={"before_turn": ["sleep 100"]})
        # Should not raise
        result = mw.before_turn("message", {})
        assert result == "message"


def test_timeout_logged_as_warning(caplog):
    import logging
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
        with caplog.at_level(logging.WARNING, logger="homie_core.middleware.external_hooks"):
            mw = ExternalHooksMiddleware(hooks={"before_turn": ["sleep 100"]})
            mw.before_turn("message", {})
    assert any("sleep 100" in r.message or "failed" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Non-existent command doesn't crash (OSError caught)
# ---------------------------------------------------------------------------

def test_nonexistent_command_does_not_crash():
    with patch("subprocess.run", side_effect=OSError("No such file or directory")):
        mw = ExternalHooksMiddleware(hooks={"before_turn": ["/nonexistent/cmd"]})
        result = mw.before_turn("message", {})
        assert result == "message"


def test_oserror_logged_as_warning(caplog):
    import logging
    with patch("subprocess.run", side_effect=OSError("No such file")):
        with caplog.at_level(logging.WARNING, logger="homie_core.middleware.external_hooks"):
            mw = ExternalHooksMiddleware(hooks={"after_turn": ["/bad/cmd"]})
            mw.after_turn("response", {})
    assert len(caplog.records) >= 1


# ---------------------------------------------------------------------------
# Multiple commands for same event — all fire
# ---------------------------------------------------------------------------

def test_multiple_commands_all_fire():
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(
            hooks={"before_turn": ["echo cmd1", "echo cmd2", "echo cmd3"]}
        )
        mw.before_turn("msg", {})
        assert mock_run.call_count == 3


def test_multiple_commands_called_with_correct_commands():
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(
            hooks={"after_turn": ["cmd_a", "cmd_b"]}
        )
        mw.after_turn("response", {})
        called_cmds = [c.args[0] for c in mock_run.call_args_list]
        assert "cmd_a" in called_cmds
        assert "cmd_b" in called_cmds


def test_multiple_commands_first_failure_does_not_prevent_second():
    results = []

    def fake_run(cmd, **kwargs):
        if cmd == "bad_cmd":
            raise OSError("fail")
        results.append(cmd)
        return MagicMock()

    with patch("subprocess.run", side_effect=fake_run):
        mw = ExternalHooksMiddleware(
            hooks={"before_turn": ["bad_cmd", "good_cmd"]}
        )
        mw.before_turn("msg", {})
    assert "good_cmd" in results


# ---------------------------------------------------------------------------
# subprocess.run called with correct kwargs
# ---------------------------------------------------------------------------

def test_subprocess_called_with_capture_output_and_text():
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(hooks={"before_turn": ["echo hi"]})
        mw.before_turn("msg", {})
        _, kwargs = mock_run.call_args
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True


def test_subprocess_called_with_timeout_5():
    with patch("subprocess.run") as mock_run:
        mw = ExternalHooksMiddleware(hooks={"before_turn": ["echo hi"]})
        mw.before_turn("msg", {})
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 5


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw = ExternalHooksMiddleware()
    assert mw.name == "external_hooks"
    assert mw.order == 95
