from __future__ import annotations

import pytest
from pathlib import Path

from homie_core.middleware.memory import MemoryMiddleware


# ---------------------------------------------------------------------------
# No memory files
# ---------------------------------------------------------------------------

def test_no_memory_files_prompt_unchanged():
    mw = MemoryMiddleware()
    prompt = "You are Homie."
    assert mw.modify_prompt(prompt) == prompt


def test_empty_memory_paths_prompt_unchanged():
    mw = MemoryMiddleware(memory_paths=[])
    prompt = "You are Homie."
    assert mw.modify_prompt(prompt) == prompt


# ---------------------------------------------------------------------------
# Valid AGENTS.md file
# ---------------------------------------------------------------------------

def test_valid_agents_md_content_injected(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("User prefers concise responses.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p])
    result = mw.modify_prompt("Base prompt.")
    assert "User prefers concise responses." in result
    assert result.startswith("Base prompt.")


def test_memory_section_header_present(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("Some memory.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p])
    result = mw.modify_prompt("Base.")
    assert "[PERSISTENT MEMORY]" in result


def test_memory_file_path_shown_as_separator(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("Content here.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p])
    result = mw.modify_prompt("Base.")
    assert str(p) in result


def test_memory_file_surrounded_by_dividers(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("Memory content.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p])
    result = mw.modify_prompt("Base.")
    assert "---" in result


# ---------------------------------------------------------------------------
# Multiple files
# ---------------------------------------------------------------------------

def test_multiple_memory_files_all_injected(tmp_path):
    p1 = tmp_path / "AGENTS.md"
    p1.write_text("Memory from file 1.", encoding="utf-8")
    p2 = tmp_path / "project_context.md"
    p2.write_text("Memory from file 2.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p1, p2])
    result = mw.modify_prompt("Base.")
    assert "Memory from file 1." in result
    assert "Memory from file 2." in result


def test_multiple_files_only_one_header(tmp_path):
    p1 = tmp_path / "a.md"
    p1.write_text("A content.", encoding="utf-8")
    p2 = tmp_path / "b.md"
    p2.write_text("B content.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p1, p2])
    result = mw.modify_prompt("Base.")
    assert result.count("[PERSISTENT MEMORY]") == 1


# ---------------------------------------------------------------------------
# Nonexistent path → skipped silently
# ---------------------------------------------------------------------------

def test_nonexistent_path_skipped_silently(tmp_path):
    nonexistent = tmp_path / "ghost.md"
    mw = MemoryMiddleware(memory_paths=[nonexistent])
    result = mw.modify_prompt("Base.")
    assert result == "Base."


def test_nonexistent_path_mixed_with_valid(tmp_path):
    valid = tmp_path / "real.md"
    valid.write_text("Real content.", encoding="utf-8")
    ghost = tmp_path / "ghost.md"
    mw = MemoryMiddleware(memory_paths=[ghost, valid])
    result = mw.modify_prompt("Base.")
    assert "Real content." in result
    assert "ghost.md" not in result


# ---------------------------------------------------------------------------
# reload() picks up file changes
# ---------------------------------------------------------------------------

def test_reload_picks_up_new_content(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("Initial memory.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p])
    result_before = mw.modify_prompt("Base.")
    assert "Initial memory." in result_before

    # Simulate agent updating the file
    p.write_text("Updated memory after reload.", encoding="utf-8")
    mw.reload()
    result_after = mw.modify_prompt("Base.")
    assert "Updated memory after reload." in result_after
    assert "Initial memory." not in result_after


def test_reload_handles_deleted_file_gracefully(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("Some memory.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p])
    # Delete the file before reloading
    p.unlink()
    # reload() should not crash
    mw.reload()


def test_reload_preserves_other_files_when_one_deleted(tmp_path):
    p1 = tmp_path / "a.md"
    p1.write_text("File A content.", encoding="utf-8")
    p2 = tmp_path / "b.md"
    p2.write_text("File B content.", encoding="utf-8")
    mw = MemoryMiddleware(memory_paths=[p1, p2])
    # Delete p1
    p1.unlink()
    mw.reload()
    result = mw.modify_prompt("Base.")
    assert "File B content." in result


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw = MemoryMiddleware()
    assert mw.name == "memory"
    assert mw.order == 12
