from __future__ import annotations

import pytest
from pathlib import Path

from homie_core.middleware.skills import SkillsMiddleware


def make_skill_file(tmp_path: Path, name: str, description: str, extra: str = "") -> Path:
    """Helper: write a SKILL.md with valid YAML frontmatter."""
    p = tmp_path / f"{name}.md"
    p.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{extra}",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# No skills
# ---------------------------------------------------------------------------

def test_no_skills_prompt_unchanged():
    mw = SkillsMiddleware()
    prompt = "You are Homie."
    assert mw.modify_prompt(prompt) == prompt


def test_empty_skill_paths_prompt_unchanged():
    mw = SkillsMiddleware(skill_paths=[])
    prompt = "You are Homie."
    assert mw.modify_prompt(prompt) == prompt


# ---------------------------------------------------------------------------
# Valid skill
# ---------------------------------------------------------------------------

def test_valid_skill_listed_in_prompt(tmp_path):
    p = make_skill_file(tmp_path, "web_search", "Search the web for information")
    mw = SkillsMiddleware(skill_paths=[p])
    result = mw.modify_prompt("Base prompt.")
    assert "web_search" in result
    assert "Search the web for information" in result
    assert result.startswith("Base prompt.")


def test_skill_entry_contains_path(tmp_path):
    p = make_skill_file(tmp_path, "read_docs", "Read documentation files")
    mw = SkillsMiddleware(skill_paths=[p])
    result = mw.modify_prompt("Base.")
    assert str(p) in result


def test_skill_listed_under_available_skills_header(tmp_path):
    p = make_skill_file(tmp_path, "summarize", "Summarize long text")
    mw = SkillsMiddleware(skill_paths=[p])
    result = mw.modify_prompt("Base.")
    assert "[AVAILABLE SKILLS]" in result


def test_prompt_contains_read_instruction(tmp_path):
    p = make_skill_file(tmp_path, "calc", "Perform calculations")
    mw = SkillsMiddleware(skill_paths=[p])
    result = mw.modify_prompt("Base.")
    assert "read" in result.lower()


# ---------------------------------------------------------------------------
# File > 10MB → skipped
# ---------------------------------------------------------------------------

def test_file_over_10mb_skipped(tmp_path, monkeypatch):
    p = make_skill_file(tmp_path, "big_skill", "A very large skill")
    # Monkeypatch stat to report a large size
    original_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        st = original_stat(self, *args, **kwargs)
        if self == p:
            import os
            # Return a stat_result with st_size = 11MB
            class FakeStat:
                st_size = 11 * 1024 * 1024
                def __getattr__(self, name):
                    return getattr(st, name)
            return FakeStat()
        return st

    monkeypatch.setattr(Path, "stat", fake_stat)
    mw = SkillsMiddleware(skill_paths=[p])
    result = mw.modify_prompt("Base.")
    assert "big_skill" not in result


# ---------------------------------------------------------------------------
# File without frontmatter → skipped
# ---------------------------------------------------------------------------

def test_file_without_frontmatter_skipped(tmp_path):
    p = tmp_path / "no_front.md"
    p.write_text("Just plain markdown, no frontmatter at all.", encoding="utf-8")
    mw = SkillsMiddleware(skill_paths=[p])
    result = mw.modify_prompt("Base.")
    assert "[AVAILABLE SKILLS]" not in result


def test_file_with_partial_frontmatter_skipped(tmp_path):
    p = tmp_path / "partial.md"
    # Has opening --- but no closing ---
    p.write_text("---\nname: incomplete\n", encoding="utf-8")
    mw = SkillsMiddleware(skill_paths=[p])
    result = mw.modify_prompt("Base.")
    assert "[AVAILABLE SKILLS]" not in result


def test_file_frontmatter_without_name_skipped(tmp_path):
    p = tmp_path / "no_name.md"
    p.write_text("---\ndescription: Some skill\n---\nContent.", encoding="utf-8")
    mw = SkillsMiddleware(skill_paths=[p])
    result = mw.modify_prompt("Base.")
    assert "[AVAILABLE SKILLS]" not in result


# ---------------------------------------------------------------------------
# Nonexistent file → skipped silently
# ---------------------------------------------------------------------------

def test_nonexistent_path_skipped_silently(tmp_path):
    nonexistent = tmp_path / "ghost.md"
    mw = SkillsMiddleware(skill_paths=[nonexistent])
    result = mw.modify_prompt("Base.")
    assert result == "Base."


# ---------------------------------------------------------------------------
# Multiple skills
# ---------------------------------------------------------------------------

def test_multiple_skills_all_listed(tmp_path):
    p1 = make_skill_file(tmp_path, "skill_a", "Does A")
    p2 = make_skill_file(tmp_path, "skill_b", "Does B")
    p3 = make_skill_file(tmp_path, "skill_c", "Does C")
    mw = SkillsMiddleware(skill_paths=[p1, p2, p3])
    result = mw.modify_prompt("Base.")
    assert "skill_a" in result
    assert "skill_b" in result
    assert "skill_c" in result
    assert "Does A" in result
    assert "Does B" in result
    assert "Does C" in result


def test_multiple_skills_only_one_header(tmp_path):
    p1 = make_skill_file(tmp_path, "alpha", "Alpha skill")
    p2 = make_skill_file(tmp_path, "beta", "Beta skill")
    mw = SkillsMiddleware(skill_paths=[p1, p2])
    result = mw.modify_prompt("Base.")
    assert result.count("[AVAILABLE SKILLS]") == 1


# ---------------------------------------------------------------------------
# Skill dict structure
# ---------------------------------------------------------------------------

def test_skill_dict_has_name_description_path(tmp_path):
    p = make_skill_file(tmp_path, "my_skill", "My skill description")
    mw = SkillsMiddleware(skill_paths=[p])
    assert len(mw._skills) == 1
    skill = mw._skills[0]
    assert skill["name"] == "my_skill"
    assert skill["description"] == "My skill description"
    assert skill["path"] == str(p)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_name_and_order():
    mw = SkillsMiddleware()
    assert mw.name == "skills"
    assert mw.order == 20
