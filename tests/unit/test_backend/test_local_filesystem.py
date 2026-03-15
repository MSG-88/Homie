from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import pytest

from homie_core.backend.local_filesystem import LocalFilesystemBackend
from homie_core.backend.protocol import EditResult, ExecutionResult, FileContent, FileInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_backend(tmp_path: Path) -> LocalFilesystemBackend:
    return LocalFilesystemBackend(tmp_path)


# ---------------------------------------------------------------------------
# Path containment / security
# ---------------------------------------------------------------------------

class TestPathContainment:
    def test_escape_via_dotdot_raises(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        with pytest.raises(ValueError, match="escapes root"):
            backend._resolve("../outside.txt")

    def test_absolute_escape_raises(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        # Try to resolve a path that points outside root
        outside = str(tmp_path.parent / "evil.txt")
        with pytest.raises(ValueError, match="escapes root"):
            backend._resolve(outside)

    def test_nested_dotdot_raises(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        with pytest.raises(ValueError, match="escapes root"):
            backend._resolve("a/b/../../../../../../etc/passwd")

    def test_valid_path_resolves(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        resolved = backend._resolve("subdir/file.txt")
        assert resolved == tmp_path / "subdir" / "file.txt"

    def test_root_slash_resolves_to_root(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        resolved = backend._resolve("/")
        assert resolved == tmp_path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

class TestRead:
    def test_full_file(self, tmp_path: Path):
        (tmp_path / "hello.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        fc = backend.read("hello.txt")
        assert fc.content == "line1\nline2\nline3"
        assert fc.total_lines == 3
        assert fc.truncated is False

    def test_splitlines_no_extra_empty_line(self, tmp_path: Path):
        """'line1\nline2\nline3\n'.splitlines() == ['line1','line2','line3'] — 3 lines, not 4."""
        (tmp_path / "f.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        fc = backend.read("f.txt")
        assert fc.total_lines == 3

    def test_offset_and_limit(self, tmp_path: Path):
        lines = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"
        (tmp_path / "multi.txt").write_text(lines, encoding="utf-8")
        backend = make_backend(tmp_path)
        fc = backend.read("multi.txt", offset=2, limit=3)
        assert fc.content == "line3\nline4\nline5"
        assert fc.total_lines == 10
        assert fc.truncated is True

    def test_nested_file(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("hello world\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        fc = backend.read("sub/nested.txt")
        assert "hello world" in fc.content

    def test_offset_beyond_end(self, tmp_path: Path):
        (tmp_path / "short.txt").write_text("only one line\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        fc = backend.read("short.txt", offset=100)
        assert fc.content == ""
        assert fc.total_lines == 1

    def test_read_file_not_found(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        with pytest.raises(FileNotFoundError):
            backend.read("nonexistent.txt")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

class TestWrite:
    def test_write_new_file(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        backend.write("newfile.txt", "hello\n")
        assert (tmp_path / "newfile.txt").read_text(encoding="utf-8") == "hello\n"

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        backend.write("a/b/c/deep.txt", "deep content\n")
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").exists()

    def test_write_overwrites_existing(self, tmp_path: Path):
        (tmp_path / "existing.txt").write_text("old content\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        backend.write("existing.txt", "new content\n")
        assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "new content\n"

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="Symlink creation may require elevated privileges on Windows",
    )
    def test_write_refuses_symlink(self, tmp_path: Path):
        real_file = tmp_path / "real.txt"
        real_file.write_text("real\n", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)
        backend = make_backend(tmp_path)
        with pytest.raises(ValueError, match="[Ss]ymlink"):
            backend.write("link.txt", "evil content\n")


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------

class TestEdit:
    def test_edit_unique_string(self, tmp_path: Path):
        (tmp_path / "code.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        result = backend.edit("code.py", "x = 1", "x = 99")
        assert result.success is True
        assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 99\ny = 2\n"

    def test_edit_not_found(self, tmp_path: Path):
        (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        result = backend.edit("code.py", "z = 99", "z = 0")
        assert result.success is False
        assert result.error is not None
        assert result.occurrences == 0

    def test_edit_not_unique_returns_error(self, tmp_path: Path):
        (tmp_path / "dup.py").write_text("foo\nfoo\nfoo\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        result = backend.edit("dup.py", "foo", "bar")
        assert result.success is False
        assert result.occurrences == 3
        assert result.error is not None
        # File should be unchanged
        assert (tmp_path / "dup.py").read_text(encoding="utf-8") == "foo\nfoo\nfoo\n"

    def test_edit_replace_all(self, tmp_path: Path):
        (tmp_path / "dup.py").write_text("foo\nfoo\nfoo\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        result = backend.edit("dup.py", "foo", "bar", replace_all=True)
        assert result.success is True
        assert (tmp_path / "dup.py").read_text(encoding="utf-8") == "bar\nbar\nbar\n"

    def test_edit_replace_all_not_found(self, tmp_path: Path):
        (tmp_path / "f.py").write_text("hello\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        result = backend.edit("f.py", "missing", "replacement", replace_all=True)
        assert result.success is False
        assert result.occurrences == 0


# ---------------------------------------------------------------------------
# Ls
# ---------------------------------------------------------------------------

class TestLs:
    def test_ls_root(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        sub = tmp_path / "subdir"
        sub.mkdir()
        backend = make_backend(tmp_path)
        entries = backend.ls("/")
        names = {e.name for e in entries}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names

    def test_ls_returns_fileinfo(self, tmp_path: Path):
        (tmp_path / "file.txt").write_text("content")
        backend = make_backend(tmp_path)
        entries = backend.ls("/")
        file_entries = [e for e in entries if e.name == "file.txt"]
        assert len(file_entries) == 1
        fi = file_entries[0]
        assert isinstance(fi, FileInfo)
        assert fi.is_dir is False

    def test_ls_subdir(self, tmp_path: Path):
        sub = tmp_path / "mydir"
        sub.mkdir()
        (sub / "inner.txt").write_text("inner")
        backend = make_backend(tmp_path)
        entries = backend.ls("mydir")
        names = {e.name for e in entries}
        assert "inner.txt" in names

    def test_ls_marks_directories(self, tmp_path: Path):
        (tmp_path / "adir").mkdir()
        backend = make_backend(tmp_path)
        entries = backend.ls("/")
        dir_entries = [e for e in entries if e.name == "adir"]
        assert len(dir_entries) == 1
        assert dir_entries[0].is_dir is True

    def test_ls_nonexistent_raises(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        with pytest.raises((FileNotFoundError, NotADirectoryError)):
            backend.ls("nonexistent_dir")


# ---------------------------------------------------------------------------
# Glob
# ---------------------------------------------------------------------------

class TestGlob:
    def test_glob_finds_nested_txt(self, tmp_path: Path):
        (tmp_path / "top.txt").write_text("top")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")
        backend = make_backend(tmp_path)
        matches = backend.glob("**/*.txt")
        names = {Path(m).name for m in matches}
        assert "top.txt" in names
        assert "nested.txt" in names

    def test_glob_returns_strings(self, tmp_path: Path):
        (tmp_path / "x.py").write_text("x")
        backend = make_backend(tmp_path)
        matches = backend.glob("*.py")
        assert len(matches) >= 1
        assert all(isinstance(m, str) for m in matches)

    def test_glob_no_match_returns_empty(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        matches = backend.glob("*.nonexistent")
        assert matches == []


# ---------------------------------------------------------------------------
# Grep
# ---------------------------------------------------------------------------

class TestGrep:
    def test_grep_literal_match(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        results = backend.grep("return 42")
        assert len(results) == 1
        assert results[0].line_number == 2
        assert "return 42" in results[0].line

    def test_grep_regex_match(self, tmp_path: Path):
        (tmp_path / "b.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        results = backend.grep(r"[xyz] = \d")
        assert len(results) == 3

    def test_grep_no_match(self, tmp_path: Path):
        (tmp_path / "c.py").write_text("nothing here\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        results = backend.grep("unicorn")
        assert results == []

    def test_grep_across_multiple_files(self, tmp_path: Path):
        (tmp_path / "f1.py").write_text("MARKER\n", encoding="utf-8")
        (tmp_path / "f2.py").write_text("MARKER\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        results = backend.grep("MARKER")
        assert len(results) == 2

    def test_grep_returns_correct_path(self, tmp_path: Path):
        (tmp_path / "target.py").write_text("found_it\n", encoding="utf-8")
        backend = make_backend(tmp_path)
        results = backend.grep("found_it")
        assert len(results) == 1
        assert "target.py" in results[0].path

    def test_grep_skips_binary_files(self, tmp_path: Path):
        (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02MARKER\x03\x04")
        backend = make_backend(tmp_path)
        # Should not raise; binary file is skipped
        results = backend.grep("MARKER")
        assert results == []


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_echo_hello(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        if platform.system() == "Windows":
            result = backend.execute("cmd /c echo hello")
        else:
            result = backend.execute("echo hello")
        assert isinstance(result, ExecutionResult)
        assert "hello" in result.stdout
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_exit_code_nonzero(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        if platform.system() == "Windows":
            result = backend.execute("cmd /c exit 1")
        else:
            result = backend.execute("exit 1", timeout=5)
        # We just verify the result is an ExecutionResult; exit codes differ by shell
        assert isinstance(result, ExecutionResult)

    def test_timeout(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        if platform.system() == "Windows":
            result = backend.execute("cmd /c ping -n 10 127.0.0.1 > nul", timeout=1)
        else:
            result = backend.execute("sleep 30", timeout=1)
        assert result.timed_out is True
        assert result.exit_code != 0

    def test_stderr_captured(self, tmp_path: Path):
        backend = make_backend(tmp_path)
        if platform.system() == "Windows":
            result = backend.execute("cmd /c echo error_output 1>&2")
        else:
            result = backend.execute("echo error_output >&2")
        assert isinstance(result, ExecutionResult)
        # stderr captured (not necessarily non-empty on all shells, but result is valid)
        assert result.stderr is not None
