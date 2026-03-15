from __future__ import annotations

import pytest

from homie_core.middleware.edit_conflict import EditConflictMiddleware
from homie_core.backend.state import StateBackend


def make_backend_and_mw() -> tuple[StateBackend, EditConflictMiddleware]:
    backend = StateBackend()
    mw = EditConflictMiddleware(backend=backend)
    return backend, mw


# ---------------------------------------------------------------------------
# Valid unique edit → passes through
# ---------------------------------------------------------------------------

def test_valid_unique_edit_passes():
    backend, mw = make_backend_and_mw()
    backend.write("/foo.py", "hello world\ngoodbye world\n")
    args = {"path": "/foo.py", "old": "hello world", "new": "hi world"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result is not None
    assert "_conflict" not in result


def test_valid_unique_edit_returns_original_args():
    backend, mw = make_backend_and_mw()
    backend.write("/foo.py", "unique string here\n")
    args = {"path": "/foo.py", "old": "unique string here", "new": "replaced"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result is args


def test_valid_unique_edit_uses_old_string_key():
    backend, mw = make_backend_and_mw()
    backend.write("/bar.py", "def foo(): pass\n")
    args = {"path": "/bar.py", "old_string": "def foo(): pass", "new": "def foo(): return 1"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result is not None


def test_valid_unique_edit_for_edit_tool_name():
    backend, mw = make_backend_and_mw()
    backend.write("/baz.py", "x = 1\n")
    args = {"path": "/baz.py", "old": "x = 1", "new": "x = 2"}
    result = mw.wrap_tool_call("edit", args)
    assert result is not None


# ---------------------------------------------------------------------------
# String not found → blocked with conflict message
# ---------------------------------------------------------------------------

def test_string_not_found_blocked():
    backend, mw = make_backend_and_mw()
    backend.write("/foo.py", "hello world\n")
    args = {"path": "/foo.py", "old": "nonexistent string", "new": "replacement"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result is None


def test_string_not_found_sets_conflict_key():
    backend, mw = make_backend_and_mw()
    backend.write("/foo.py", "hello world\n")
    args = {"path": "/foo.py", "old": "nonexistent string", "new": "replacement"}
    mw.wrap_tool_call("edit_file", args)
    assert "_conflict" in args
    assert "not found" in args["_conflict"].lower() or "String not found" in args["_conflict"]


def test_string_not_found_conflict_mentions_path():
    backend, mw = make_backend_and_mw()
    backend.write("/myfile.py", "some content\n")
    args = {"path": "/myfile.py", "old": "missing", "new": "x"}
    mw.wrap_tool_call("edit_file", args)
    assert "/myfile.py" in args["_conflict"]


# ---------------------------------------------------------------------------
# String found multiple times without replace_all → blocked
# ---------------------------------------------------------------------------

def test_duplicate_string_without_replace_all_blocked():
    backend, mw = make_backend_and_mw()
    backend.write("/dup.py", "foo\nfoo\nbar\n")
    args = {"path": "/dup.py", "old": "foo", "new": "baz"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result is None


def test_duplicate_string_conflict_message_mentions_count():
    backend, mw = make_backend_and_mw()
    backend.write("/dup.py", "x\nx\nx\n")
    args = {"path": "/dup.py", "old": "x", "new": "y"}
    mw.wrap_tool_call("edit_file", args)
    assert "_conflict" in args
    assert "3" in args["_conflict"]


def test_duplicate_string_conflict_message_suggests_replace_all():
    backend, mw = make_backend_and_mw()
    backend.write("/dup.py", "dup\ndup\n")
    args = {"path": "/dup.py", "old": "dup", "new": "uniq"}
    mw.wrap_tool_call("edit_file", args)
    assert "replace_all" in args["_conflict"]


# ---------------------------------------------------------------------------
# String found multiple times with replace_all → passes
# ---------------------------------------------------------------------------

def test_duplicate_string_with_replace_all_passes():
    backend, mw = make_backend_and_mw()
    backend.write("/dup.py", "foo\nfoo\n")
    args = {"path": "/dup.py", "old": "foo", "new": "bar", "replace_all": True}
    result = mw.wrap_tool_call("edit_file", args)
    assert result is not None
    assert "_conflict" not in result


def test_duplicate_string_with_replace_all_returns_args():
    backend, mw = make_backend_and_mw()
    backend.write("/dup.py", "a\na\na\n")
    args = {"path": "/dup.py", "old": "a", "new": "b", "replace_all": True}
    result = mw.wrap_tool_call("edit_file", args)
    assert result is args


# ---------------------------------------------------------------------------
# Non-edit tools → passthrough
# ---------------------------------------------------------------------------

def test_non_edit_tool_passthrough():
    backend, mw = make_backend_and_mw()
    args = {"command": "git status"}
    result = mw.wrap_tool_call("run_command", args)
    assert result == args


def test_read_file_passthrough():
    backend, mw = make_backend_and_mw()
    args = {"path": "/foo.py"}
    result = mw.wrap_tool_call("read_file", args)
    assert result == args


def test_write_file_passthrough():
    backend, mw = make_backend_and_mw()
    args = {"path": "/foo.py", "content": "hello"}
    result = mw.wrap_tool_call("write_file", args)
    assert result == args


def test_search_passthrough():
    backend, mw = make_backend_and_mw()
    args = {"query": "foo"}
    result = mw.wrap_tool_call("search", args)
    assert result == args


# ---------------------------------------------------------------------------
# File not found → blocked
# ---------------------------------------------------------------------------

def test_file_not_found_blocked():
    backend, mw = make_backend_and_mw()
    # No file written — file doesn't exist
    args = {"path": "/nonexistent.py", "old": "anything", "new": "something"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result is None


def test_file_not_found_conflict_message():
    backend, mw = make_backend_and_mw()
    args = {"path": "/missing.py", "old": "code", "new": "replaced"}
    mw.wrap_tool_call("edit_file", args)
    assert "_conflict" in args
    assert "not found" in args["_conflict"].lower() or "File not found" in args["_conflict"]


def test_file_not_found_conflict_mentions_path():
    backend, mw = make_backend_and_mw()
    args = {"path": "/missing/deep/file.py", "old": "x", "new": "y"}
    mw.wrap_tool_call("edit_file", args)
    assert "/missing/deep/file.py" in args["_conflict"]


# ---------------------------------------------------------------------------
# Missing path/old args → passthrough (let tool handle)
# ---------------------------------------------------------------------------

def test_missing_path_passthrough():
    backend, mw = make_backend_and_mw()
    args = {"old": "something", "new": "replacement"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result == args


def test_missing_old_passthrough():
    backend, mw = make_backend_and_mw()
    args = {"path": "/foo.py", "new": "replacement"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result == args


def test_empty_path_passthrough():
    backend, mw = make_backend_and_mw()
    args = {"path": "", "old": "something", "new": "replacement"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result == args


def test_empty_old_passthrough():
    backend, mw = make_backend_and_mw()
    args = {"path": "/foo.py", "old": "", "new": "replacement"}
    result = mw.wrap_tool_call("edit_file", args)
    assert result == args


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    backend, mw = make_backend_and_mw()
    assert mw.name == "edit_conflict"
    assert mw.order == 80
