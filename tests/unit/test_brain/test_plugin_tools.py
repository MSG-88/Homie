"""Tests for plugin tools: git, terminal, clipboard, notes, and web search."""
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from homie_core.brain.tool_registry import ToolCall, ToolRegistry
from homie_core.brain.builtin_tools import register_builtin_tools
from homie_core.memory.working import WorkingMemory


@pytest.fixture
def registry(tmp_path):
    """Create a registry with all tools registered, using tmp_path for storage."""
    reg = ToolRegistry()
    wm = WorkingMemory()
    register_builtin_tools(
        registry=reg,
        working_memory=wm,
        storage_path=str(tmp_path),
    )
    return reg


# -----------------------------------------------------------------------
# Git tools
# -----------------------------------------------------------------------

class TestGitStatus:
    def test_git_status_success(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="## main\n M file.py\n", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="git_status"))
        assert result.success is True
        assert "main" in result.output
        assert "file.py" in result.output

    def test_git_status_clean(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="git_status"))
        assert result.success is True
        assert "clean" in result.output.lower()

    def test_git_status_error(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="fatal: not a git repo")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="git_status"))
        assert "error" in result.output.lower() or "fatal" in result.output.lower()

    def test_git_status_not_installed(self, registry):
        with patch("homie_core.brain.builtin_tools.subprocess.run", side_effect=FileNotFoundError):
            result = registry.execute(ToolCall(name="git_status"))
        assert "not installed" in result.output.lower()

    def test_git_status_timeout(self, registry):
        with patch("homie_core.brain.builtin_tools.subprocess.run", side_effect=subprocess.TimeoutExpired("git", 10)):
            result = registry.execute(ToolCall(name="git_status"))
        assert "timed out" in result.output.lower()


class TestGitLog:
    def test_git_log_default(self, registry):
        fake = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="abc123 feat: add tools\ndef456 fix: bug\n", stderr="",
        )
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="git_log"))
        assert result.success is True
        assert "abc123" in result.output

    def test_git_log_custom_count(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123 commit\n", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake) as mock_run:
            registry.execute(ToolCall(name="git_log", arguments={"count": 3}))
        # Verify -3 was passed
        call_args = mock_run.call_args[0][0]
        assert "-3" in call_args

    def test_git_log_empty(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="git_log"))
        assert "no commits" in result.output.lower()


class TestGitDiff:
    def test_git_diff_unstaged(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="diff --git a/f.py\n+hello\n", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake) as mock_run:
            result = registry.execute(ToolCall(name="git_diff"))
        assert result.success is True
        assert "+hello" in result.output
        call_args = mock_run.call_args[0][0]
        assert "--cached" not in call_args

    def test_git_diff_staged(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="staged diff\n", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake) as mock_run:
            result = registry.execute(ToolCall(name="git_diff", arguments={"staged": True}))
        call_args = mock_run.call_args[0][0]
        assert "--cached" in call_args

    def test_git_diff_no_changes(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="git_diff"))
        assert "no changes" in result.output.lower()

    def test_git_diff_truncation(self, registry):
        big_output = "x" * 6000
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=big_output, stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="git_diff"))
        assert "truncated" in result.output
        assert len(result.output) < 6000


class TestGitBranch:
    def test_git_branch_list(self, registry):
        fake = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="* main\n  develop\n  feature/x\n", stderr="",
        )
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="git_branch"))
        assert result.success is True
        assert "* main" in result.output
        assert "develop" in result.output


# -----------------------------------------------------------------------
# Terminal tool
# -----------------------------------------------------------------------

class TestRunCommand:
    def test_safe_command(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="hello world\n", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="run_command", arguments={"command": "echo hello"}))
        assert result.success is True
        assert "hello world" in result.output

    def test_blocked_rm_rf_root(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "rm -rf /"}))
        assert "Blocked" in result.output

    def test_blocked_rm_rf_home(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "rm -rf ~"}))
        assert "Blocked" in result.output

    def test_blocked_format(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "format C:"}))
        assert "Blocked" in result.output

    def test_blocked_shutdown(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "shutdown -h now"}))
        assert "Blocked" in result.output

    def test_blocked_reboot(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "reboot"}))
        assert "Blocked" in result.output

    def test_blocked_mkfs(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "mkfs.ext4 /dev/sda"}))
        assert "Blocked" in result.output

    def test_blocked_dd(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "dd if=/dev/zero of=/dev/sda"}))
        assert "Blocked" in result.output

    def test_blocked_fork_bomb(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": ":(){ :|:& };:"}))
        assert "Blocked" in result.output

    def test_blocked_chmod_recursive_root(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "chmod -R 777 /"}))
        assert "Blocked" in result.output

    def test_blocked_del_windows(self, registry):
        result = registry.execute(ToolCall(name="run_command", arguments={"command": "del /f /s /q C:\\"}))
        assert "Blocked" in result.output

    def test_command_timeout(self, registry):
        with patch("homie_core.brain.builtin_tools.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
            result = registry.execute(ToolCall(name="run_command", arguments={"command": "sleep 100"}))
        assert "timed out" in result.output.lower()

    def test_output_truncation(self, registry):
        big_output = "x" * 6000
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=big_output, stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="run_command", arguments={"command": "cat bigfile"}))
        assert "truncated" in result.output
        assert len(result.output) < 6000

    def test_empty_output_shows_exit_code(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="run_command", arguments={"command": "true"}))
        assert "exit code 0" in result.output.lower()

    def test_stderr_included(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error: not found\n")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="run_command", arguments={"command": "bad_cmd"}))
        assert "not found" in result.output


# -----------------------------------------------------------------------
# Clipboard tools
# -----------------------------------------------------------------------

class TestClipboardRead:
    def test_clipboard_read_success(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="clipboard text\n", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="clipboard_read"))
        assert result.success is True
        assert "clipboard text" in result.output

    def test_clipboard_read_empty(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(name="clipboard_read"))
        assert "empty" in result.output.lower()

    def test_clipboard_read_not_available(self, registry):
        with patch("homie_core.brain.builtin_tools.subprocess.run", side_effect=FileNotFoundError):
            result = registry.execute(ToolCall(name="clipboard_read"))
        assert "not available" in result.output.lower()


class TestClipboardWrite:
    def test_clipboard_write_success(self, registry):
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("homie_core.brain.builtin_tools.subprocess.run", return_value=fake):
            result = registry.execute(ToolCall(
                name="clipboard_write", arguments={"text": "hello world"},
            ))
        assert result.success is True
        assert "Copied" in result.output
        assert "11 chars" in result.output

    def test_clipboard_write_not_available(self, registry):
        with patch("homie_core.brain.builtin_tools.subprocess.run", side_effect=FileNotFoundError):
            result = registry.execute(ToolCall(
                name="clipboard_write", arguments={"text": "test"},
            ))
        assert "not available" in result.output.lower()


# -----------------------------------------------------------------------
# Notes tools
# -----------------------------------------------------------------------

class TestSaveNote:
    def test_save_note(self, registry, tmp_path):
        result = registry.execute(ToolCall(
            name="save_note",
            arguments={"title": "My Note", "content": "Some important content."},
        ))
        assert result.success is True
        assert "saved" in result.output.lower()
        # Verify file exists
        note_file = tmp_path / "notes" / "My_Note.md"
        assert note_file.exists()
        text = note_file.read_text(encoding="utf-8")
        assert "# My Note" in text
        assert "Some important content." in text

    def test_save_note_special_chars(self, registry, tmp_path):
        result = registry.execute(ToolCall(
            name="save_note",
            arguments={"title": "TODO: fix bug #123!", "content": "Fix it."},
        ))
        assert result.success is True
        assert "saved" in result.output.lower()

    def test_save_note_empty_title(self, registry):
        # Title with only spaces/special chars that strip to empty
        result = registry.execute(ToolCall(
            name="save_note",
            arguments={"title": "   ", "content": "test"},
        ))
        assert result.success is True
        assert "Invalid" in result.output


class TestListNotes:
    def test_list_notes_empty(self, registry):
        result = registry.execute(ToolCall(name="list_notes"))
        assert result.success is True
        assert "no notes" in result.output.lower()

    def test_list_notes_with_notes(self, registry, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "note1.md").write_text("# Note 1\ncontent", encoding="utf-8")
        (notes_dir / "note2.md").write_text("# Note 2\ncontent", encoding="utf-8")

        result = registry.execute(ToolCall(name="list_notes"))
        assert result.success is True
        assert "2 notes" in result.output
        assert "note1" in result.output
        assert "note2" in result.output


class TestReadNote:
    def test_read_note_success(self, registry, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / "My_Note.md").write_text("# My Note\n\nContent here.", encoding="utf-8")

        result = registry.execute(ToolCall(
            name="read_note", arguments={"title": "My Note"},
        ))
        assert result.success is True
        assert "Content here" in result.output

    def test_read_note_not_found(self, registry, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()

        result = registry.execute(ToolCall(
            name="read_note", arguments={"title": "nonexistent"},
        ))
        assert "not found" in result.output.lower()

    def test_save_then_read_roundtrip(self, registry, tmp_path):
        registry.execute(ToolCall(
            name="save_note",
            arguments={"title": "Roundtrip", "content": "This should persist."},
        ))
        result = registry.execute(ToolCall(
            name="read_note", arguments={"title": "Roundtrip"},
        ))
        assert "This should persist" in result.output


# -----------------------------------------------------------------------
# Web search tool
# -----------------------------------------------------------------------

class TestWebSearch:
    def test_web_search_with_abstract(self, registry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "Python is a programming language.",
            "Heading": "Python",
            "AbstractURL": "https://python.org",
            "Answer": "",
            "RelatedTopics": [],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = registry.execute(ToolCall(
                name="web_search", arguments={"query": "Python programming"},
            ))
        assert result.success is True
        assert "Python" in result.output
        assert "programming language" in result.output

    def test_web_search_with_answer(self, registry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "",
            "Heading": "",
            "Answer": "42",
            "RelatedTopics": [],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = registry.execute(ToolCall(
                name="web_search", arguments={"query": "meaning of life"},
            ))
        assert "42" in result.output

    def test_web_search_with_related(self, registry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "",
            "Heading": "",
            "Answer": "",
            "RelatedTopics": [
                {"Text": "Related topic one about something interesting"},
                {"Text": "Related topic two about something else"},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = registry.execute(ToolCall(
                name="web_search", arguments={"query": "test query"},
            ))
        assert "Related" in result.output
        assert "topic one" in result.output

    def test_web_search_no_results(self, registry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Abstract": "",
            "Heading": "",
            "Answer": "",
            "RelatedTopics": [],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = registry.execute(ToolCall(
                name="web_search", arguments={"query": "xyzzy123nonsense"},
            ))
        assert "no instant results" in result.output.lower()

    def test_web_search_network_error(self, registry):
        with patch("requests.get", side_effect=Exception("Connection refused")):
            result = registry.execute(ToolCall(
                name="web_search", arguments={"query": "test"},
            ))
        assert "error" in result.output.lower()

    def test_web_search_requests_not_installed(self, registry):
        import sys
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("No module named 'requests'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = registry.execute(ToolCall(
                name="web_search", arguments={"query": "test"},
            ))
        assert result.success is True
        assert "unavailable" in result.output.lower()


# -----------------------------------------------------------------------
# Registration check
# -----------------------------------------------------------------------

class TestPluginToolsRegistered:
    def test_all_plugin_tools_exist(self, registry):
        expected = [
            "git_status", "git_log", "git_diff", "git_branch",
            "run_command",
            "clipboard_read", "clipboard_write",
            "save_note", "list_notes", "read_note",
            "web_search",
        ]
        for name in expected:
            assert registry.get(name) is not None, f"Tool '{name}' not registered"

    def test_tool_categories(self, registry):
        assert registry.get("git_status").category == "git"
        assert registry.get("git_log").category == "git"
        assert registry.get("git_diff").category == "git"
        assert registry.get("git_branch").category == "git"
        assert registry.get("run_command").category == "system"
        assert registry.get("clipboard_read").category == "clipboard"
        assert registry.get("clipboard_write").category == "clipboard"
        assert registry.get("save_note").category == "notes"
        assert registry.get("list_notes").category == "notes"
        assert registry.get("read_note").category == "notes"
        assert registry.get("web_search").category == "web"
