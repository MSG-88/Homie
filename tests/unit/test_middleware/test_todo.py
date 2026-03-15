from __future__ import annotations

import json
import pytest

from homie_core.middleware.todo import TodoMiddleware


def make_mw() -> TodoMiddleware:
    return TodoMiddleware()


# ---------------------------------------------------------------------------
# modify_tools
# ---------------------------------------------------------------------------

def test_modify_tools_adds_write_todos():
    mw = make_mw()
    tools = mw.modify_tools([])
    names = [t["name"] for t in tools]
    assert "write_todos" in names


def test_modify_tools_preserves_existing_tools():
    mw = make_mw()
    existing = [{"name": "search"}, {"name": "read_file"}]
    tools = mw.modify_tools(existing)
    names = [t["name"] for t in tools]
    assert "search" in names
    assert "read_file" in names
    assert "write_todos" in names
    assert len(tools) == 3


def test_write_todos_tool_has_description():
    mw = make_mw()
    tools = mw.modify_tools([])
    todo_tool = next(t for t in tools if t["name"] == "write_todos")
    assert "description" in todo_tool
    assert len(todo_tool["description"]) > 0


# ---------------------------------------------------------------------------
# modify_prompt — no todos
# ---------------------------------------------------------------------------

def test_modify_prompt_no_todos_unchanged():
    mw = make_mw()
    prompt = "You are Homie."
    assert mw.modify_prompt(prompt) == prompt


def test_modify_prompt_empty_todos_unchanged():
    mw = make_mw()
    prompt = "System prompt here."
    result = mw.modify_prompt(prompt)
    assert result == prompt


# ---------------------------------------------------------------------------
# modify_prompt — with todos
# ---------------------------------------------------------------------------

def test_modify_prompt_appends_task_block():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "Step one", "status": "pending"}]})
    result = mw.modify_prompt("Base prompt.")
    assert "[CURRENT TASKS]" in result
    assert "Step one" in result
    assert result.startswith("Base prompt.")


def test_modify_prompt_pending_status_icon():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "Do A", "status": "pending"}]})
    result = mw.modify_prompt("P.")
    assert "[ ]" in result
    assert "Do A" in result


def test_modify_prompt_in_progress_status_icon():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "Do B", "status": "in_progress"}]})
    result = mw.modify_prompt("P.")
    assert "[~]" in result
    assert "Do B" in result


def test_modify_prompt_done_status_icon():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "Do C", "status": "done"}]})
    result = mw.modify_prompt("P.")
    assert "[x]" in result
    assert "Do C" in result


def test_modify_prompt_unknown_status_defaults_to_pending_icon():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "Do D", "status": "weird_status"}]})
    result = mw.modify_prompt("P.")
    assert "[ ]" in result


def test_modify_prompt_multiple_todos_all_listed():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [
        {"task": "Alpha", "status": "done"},
        {"task": "Beta", "status": "in_progress"},
        {"task": "Gamma", "status": "pending"},
    ]})
    result = mw.modify_prompt("P.")
    assert "Alpha" in result
    assert "Beta" in result
    assert "Gamma" in result
    assert "[x]" in result
    assert "[~]" in result
    assert "[ ]" in result


def test_modify_prompt_todos_numbered():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [
        {"task": "First", "status": "pending"},
        {"task": "Second", "status": "pending"},
    ]})
    result = mw.modify_prompt("P.")
    assert "1." in result
    assert "2." in result


# ---------------------------------------------------------------------------
# wrap_tool_call — write_todos
# ---------------------------------------------------------------------------

def test_wrap_tool_call_write_todos_updates_state():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [
        {"task": "Task A", "status": "pending"},
    ]})
    assert len(mw.todos) == 1
    assert mw.todos[0]["task"] == "Task A"
    assert mw.todos[0]["status"] == "pending"


def test_wrap_tool_call_write_todos_replaces_existing():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "Old", "status": "pending"}]})
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "New", "status": "done"}]})
    assert len(mw.todos) == 1
    assert mw.todos[0]["task"] == "New"


def test_wrap_tool_call_write_todos_accepts_json_string():
    mw = make_mw()
    todos_json = json.dumps([{"task": "From JSON", "status": "in_progress"}])
    mw.wrap_tool_call("write_todos", {"todos": todos_json})
    assert len(mw.todos) == 1
    assert mw.todos[0]["task"] == "From JSON"


def test_wrap_tool_call_write_todos_invalid_json_passthrough():
    mw = make_mw()
    result = mw.wrap_tool_call("write_todos", {"todos": "not valid json {"})
    # Should return args unchanged without crashing
    assert result is not None


def test_wrap_tool_call_write_todos_empty_list_clears_todos():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "X", "status": "pending"}]})
    mw.wrap_tool_call("write_todos", {"todos": []})
    assert mw.todos == []


# ---------------------------------------------------------------------------
# wrap_tool_call — other tools (passthrough)
# ---------------------------------------------------------------------------

def test_wrap_tool_call_other_tool_passthrough():
    mw = make_mw()
    args = {"query": "search term", "limit": 10}
    result = mw.wrap_tool_call("search", args)
    assert result == args


def test_wrap_tool_call_other_tool_does_not_modify_todos():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "Keep me", "status": "pending"}]})
    mw.wrap_tool_call("search", {"query": "foo"})
    assert len(mw.todos) == 1
    assert mw.todos[0]["task"] == "Keep me"


# ---------------------------------------------------------------------------
# todos property
# ---------------------------------------------------------------------------

def test_todos_property_returns_copy():
    mw = make_mw()
    mw.wrap_tool_call("write_todos", {"todos": [{"task": "T", "status": "pending"}]})
    todos1 = mw.todos
    todos1.append({"task": "injected", "status": "pending"})
    # Internal state should not be modified
    assert len(mw.todos) == 1


def test_todos_property_initially_empty():
    mw = make_mw()
    assert mw.todos == []


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw = make_mw()
    assert mw.name == "todo"
    assert mw.order == 15
