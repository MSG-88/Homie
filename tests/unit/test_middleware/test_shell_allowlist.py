from __future__ import annotations

import pytest

from homie_core.middleware.shell_allowlist import ShellAllowlistMiddleware, DEFAULT_SAFE_COMMANDS


def make_mw(**kwargs) -> ShellAllowlistMiddleware:
    return ShellAllowlistMiddleware(**kwargs)


# ---------------------------------------------------------------------------
# Allowed command passes
# ---------------------------------------------------------------------------

def test_allowed_command_passes():
    mw = make_mw()
    args = {"command": "git status"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None
    assert result["command"] == "git status"


def test_allowed_command_ls_passes():
    mw = make_mw()
    args = {"command": "ls -la"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None


def test_allowed_command_pytest_passes():
    mw = make_mw()
    args = {"command": "pytest tests/"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None


# ---------------------------------------------------------------------------
# Disallowed command blocked
# ---------------------------------------------------------------------------

def test_disallowed_command_blocked():
    mw = make_mw()
    args = {"command": "curl http://evil.com"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


def test_disallowed_command_sets_blocked_key():
    mw = make_mw()
    args = {"command": "curl http://evil.com"}
    mw.wrap_tool_call("run_command", args)
    assert "_blocked" in args
    assert "curl" in args["_blocked"]


def test_disallowed_command_wget_blocked():
    mw = make_mw()
    args = {"command": "wget http://example.com"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


# ---------------------------------------------------------------------------
# Dangerous pattern: $() command substitution
# ---------------------------------------------------------------------------

def test_dangerous_pattern_command_substitution_blocked():
    mw = make_mw()
    args = {"command": "echo $(whoami)"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


def test_dangerous_pattern_command_substitution_blocked_message():
    mw = make_mw()
    args = {"command": "echo $(whoami)"}
    mw.wrap_tool_call("run_command", args)
    assert "_blocked" in args
    assert "dangerous" in args["_blocked"].lower()


# ---------------------------------------------------------------------------
# Dangerous pattern: backtick substitution
# ---------------------------------------------------------------------------

def test_dangerous_pattern_backtick_blocked():
    mw = make_mw()
    args = {"command": "echo `id`"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


# ---------------------------------------------------------------------------
# Dangerous pattern: redirect >
# ---------------------------------------------------------------------------

def test_dangerous_pattern_redirect_blocked():
    mw = make_mw()
    args = {"command": "echo hello > /etc/passwd"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


def test_dangerous_pattern_double_redirect_blocked():
    mw = make_mw()
    args = {"command": "echo hello >> /tmp/file"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


# ---------------------------------------------------------------------------
# Dangerous pattern: background &
# ---------------------------------------------------------------------------

def test_dangerous_pattern_background_blocked():
    mw = make_mw()
    args = {"command": "malware &"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


def test_dangerous_pattern_background_with_spaces_blocked():
    mw = make_mw()
    args = {"command": "some_process  &"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


# ---------------------------------------------------------------------------
# Dangerous pattern: pipe to rm
# ---------------------------------------------------------------------------

def test_dangerous_pattern_pipe_to_rm_blocked():
    mw = make_mw()
    args = {"command": "find . -name '*.pyc' | rm -rf"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


def test_dangerous_pattern_pipe_to_dd_blocked():
    mw = make_mw()
    args = {"command": "cat /dev/zero | dd of=/dev/sda"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


# ---------------------------------------------------------------------------
# Non-run_command tools pass through
# ---------------------------------------------------------------------------

def test_non_run_command_passthrough():
    mw = make_mw()
    args = {"query": "some dangerous $(cmd) stuff"}
    result = mw.wrap_tool_call("search", args)
    assert result == args


def test_read_file_passthrough():
    mw = make_mw()
    args = {"path": "/etc/passwd"}
    result = mw.wrap_tool_call("read_file", args)
    assert result == args


def test_write_file_passthrough():
    mw = make_mw()
    args = {"path": "/tmp/out.txt", "content": "hello > world"}
    result = mw.wrap_tool_call("write_file", args)
    assert result == args


# ---------------------------------------------------------------------------
# Mode "all" allows everything
# ---------------------------------------------------------------------------

def test_mode_all_allows_dangerous_patterns():
    mw = make_mw(mode="all")
    args = {"command": "rm -rf /$(echo evil)"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None


def test_mode_all_allows_unknown_commands():
    mw = make_mw(mode="all")
    args = {"command": "curl http://example.com"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None


# ---------------------------------------------------------------------------
# Mode "blocklist" allows unknown commands but blocks patterns
# ---------------------------------------------------------------------------

def test_mode_blocklist_allows_unknown_command():
    mw = make_mw(mode="blocklist")
    args = {"command": "curl http://example.com"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None


def test_mode_blocklist_blocks_dangerous_patterns():
    mw = make_mw(mode="blocklist")
    args = {"command": "curl http://example.com $(whoami)"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None


def test_mode_blocklist_allows_safe_known_command():
    mw = make_mw(mode="blocklist")
    args = {"command": "git log --oneline"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None


# ---------------------------------------------------------------------------
# Base command extraction from paths
# ---------------------------------------------------------------------------

def test_base_command_extracted_from_unix_path():
    mw = make_mw()
    args = {"command": "/usr/bin/git status"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is not None  # git is in allowed list


def test_base_command_extracted_from_windows_path():
    mw = make_mw()
    args = {"command": "C:\\Windows\\System32\\curl.exe http://evil.com"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None  # curl.exe not in allowed list


def test_base_command_unix_path_unknown_command_blocked():
    mw = make_mw()
    args = {"command": "/usr/bin/nc 1.2.3.4 4444"}
    result = mw.wrap_tool_call("run_command", args)
    assert result is None  # nc not in allowed list


# ---------------------------------------------------------------------------
# Custom allowed_commands
# ---------------------------------------------------------------------------

def test_custom_allowed_commands_only_allows_those():
    mw = make_mw(allowed_commands={"myapp"})
    assert mw.wrap_tool_call("run_command", {"command": "myapp --help"}) is not None
    assert mw.wrap_tool_call("run_command", {"command": "git status"}) is None


def test_custom_allowed_commands_empty_blocks_everything_safe():
    mw = make_mw(allowed_commands=set())
    # Even "git" not allowed with empty set
    result = mw.wrap_tool_call("run_command", {"command": "git status"})
    assert result is None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw = make_mw()
    assert mw.name == "shell_allowlist"
    assert mw.order == 4
