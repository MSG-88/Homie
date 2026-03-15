from __future__ import annotations

import pytest

from homie_core.backend.composite import CompositeBackend
from homie_core.backend.state import StateBackend
from homie_core.backend.protocol import FileInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_default() -> StateBackend:
    b = StateBackend()
    return b


def make_vault() -> StateBackend:
    return StateBackend()


def make_tmp() -> StateBackend:
    return StateBackend()


# ---------------------------------------------------------------------------
# Routing — read
# ---------------------------------------------------------------------------

class TestReadRouting:
    def test_read_from_default_backend(self):
        default = make_default()
        default.write("/notes.txt", "default content\n")
        cb = CompositeBackend(default=default)
        fc = cb.read("/notes.txt")
        assert "default content" in fc.content

    def test_read_from_vault_route(self):
        default = make_default()
        vault = make_vault()
        vault.write("/secret.txt", "vault content\n")
        cb = CompositeBackend(default=default, routes={"/vault": vault})
        fc = cb.read("/vault/secret.txt")
        assert "vault content" in fc.content

    def test_unmatched_path_goes_to_default(self):
        default = make_default()
        default.write("/fallback.txt", "here\n")
        cb = CompositeBackend(default=default, routes={"/other": make_tmp()})
        fc = cb.read("/fallback.txt")
        assert "here" in fc.content

    def test_longer_prefix_wins_over_shorter(self):
        default = make_default()
        short = StateBackend()
        long_ = StateBackend()
        short.write("/a/b/file.txt", "short match\n")
        long_.write("/b/file.txt", "long match\n")
        cb = CompositeBackend(
            default=default,
            routes={
                "/vault": short,
                "/vault/deep": long_,
            },
        )
        # "/vault/deep/b/file.txt" should route to *long_* ("/vault/deep" prefix)
        long_.write("/b/file.txt", "long match\n")
        fc = cb.read("/vault/deep/b/file.txt")
        assert "long match" in fc.content

    def test_read_missing_file_raises(self):
        default = make_default()
        cb = CompositeBackend(default=default)
        with pytest.raises(FileNotFoundError):
            cb.read("/nonexistent.txt")


# ---------------------------------------------------------------------------
# Routing — write
# ---------------------------------------------------------------------------

class TestWriteRouting:
    def test_write_to_tmp_route(self):
        default = make_default()
        tmp = make_tmp()
        cb = CompositeBackend(default=default, routes={"/tmp": tmp})
        cb.write("/tmp/scratch.txt", "scratch content\n")
        # Read back through the composite to confirm routing
        fc = cb.read("/tmp/scratch.txt")
        assert "scratch content" in fc.content

    def test_write_to_default_when_no_route_matches(self):
        default = make_default()
        cb = CompositeBackend(default=default, routes={"/other": make_tmp()})
        cb.write("/general.txt", "general\n")
        fc = default.read("/general.txt")
        assert "general" in fc.content


# ---------------------------------------------------------------------------
# Routing — edit
# ---------------------------------------------------------------------------

class TestEditRouting:
    def test_edit_in_routed_backend(self):
        default = make_default()
        vault = make_vault()
        vault.write("/doc.txt", "original\n")
        cb = CompositeBackend(default=default, routes={"/vault": vault})
        result = cb.edit("/vault/doc.txt", "original", "updated")
        assert result.success is True
        fc = vault.read("/doc.txt")
        assert "updated" in fc.content


# ---------------------------------------------------------------------------
# Ls root aggregation
# ---------------------------------------------------------------------------

class TestLsRoot:
    def test_ls_root_shows_route_directories(self):
        default = make_default()
        vault = make_vault()
        tmp = make_tmp()
        cb = CompositeBackend(
            default=default,
            routes={"/vault": vault, "/tmp": tmp},
        )
        entries = cb.ls("/")
        names = {e.name for e in entries}
        assert "vault" in names
        assert "tmp" in names

    def test_ls_root_route_dirs_are_marked_is_dir(self):
        default = make_default()
        cb = CompositeBackend(default=default, routes={"/vault": make_vault()})
        entries = cb.ls("/")
        vault_entries = [e for e in entries if e.name == "vault"]
        assert len(vault_entries) == 1
        assert vault_entries[0].is_dir is True

    def test_ls_root_includes_default_entries(self):
        default = make_default()
        default.write("/readme.txt", "hi")
        cb = CompositeBackend(default=default, routes={"/vault": make_vault()})
        entries = cb.ls("/")
        names = {e.name for e in entries}
        assert "readme.txt" in names

    def test_ls_root_no_duplicate_route_dirs(self):
        default = make_default()
        cb = CompositeBackend(default=default, routes={"/vault": make_vault()})
        entries = cb.ls("/")
        vault_entries = [e for e in entries if e.name == "vault"]
        assert len(vault_entries) == 1

    def test_ls_subdir_delegates_to_route(self):
        default = make_default()
        vault = make_vault()
        vault.write("/secret.txt", "s")
        cb = CompositeBackend(default=default, routes={"/vault": vault})
        entries = cb.ls("/vault")
        names = {e.name for e in entries}
        assert "secret.txt" in names


# ---------------------------------------------------------------------------
# Glob across all backends
# ---------------------------------------------------------------------------

class TestGlob:
    def test_glob_merges_all_backends(self):
        default = make_default()
        vault = make_vault()
        default.write("/a.txt", "a")
        vault.write("/b.txt", "b")
        cb = CompositeBackend(default=default, routes={"/vault": vault})
        matches = cb.glob("*.txt")
        # Both backends should contribute matches
        assert len(matches) >= 2

    def test_glob_results_are_strings(self):
        default = make_default()
        default.write("/x.py", "x")
        cb = CompositeBackend(default=default)
        matches = cb.glob("*.py")
        assert all(isinstance(m, str) for m in matches)


# ---------------------------------------------------------------------------
# Grep across all backends
# ---------------------------------------------------------------------------

class TestGrep:
    def test_grep_root_searches_all_backends(self):
        default = make_default()
        vault = make_vault()
        default.write("/a.py", "MARKER\n")
        vault.write("/b.py", "MARKER\n")
        cb = CompositeBackend(default=default, routes={"/vault": vault})
        results = cb.grep("MARKER")
        assert len(results) == 2

    def test_grep_scoped_to_route(self):
        default = make_default()
        vault = make_vault()
        default.write("/a.py", "MARKER\n")
        vault.write("/b.py", "MARKER\n")
        cb = CompositeBackend(default=default, routes={"/vault": vault})
        results = cb.grep("MARKER", path="/vault")
        assert len(results) == 1

    def test_grep_results_have_prefixed_paths(self):
        default = make_default()
        vault = make_vault()
        vault.write("/secret.py", "MARKER\n")
        cb = CompositeBackend(default=default, routes={"/vault": vault})
        results = cb.grep("MARKER")
        assert any("/vault" in r.path for r in results)
