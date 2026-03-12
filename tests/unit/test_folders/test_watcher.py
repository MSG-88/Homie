"""Tests for FolderWatcher — scanning and change detection."""
from __future__ import annotations

import os
import time
from pathlib import Path

from homie_core.folders.models import FolderWatch
from homie_core.folders.watcher import FolderWatcher
from homie_core.vault.schema import create_cache_db


def _make_watch(path: str, **overrides) -> FolderWatch:
    defaults = dict(
        id=1, path=path, label=None, scan_interval=300,
        last_scanned=None, file_count=0, enabled=True,
    )
    defaults.update(overrides)
    return FolderWatch(**defaults)


class TestFolderWatcherScan:
    def test_scan_empty_folder(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        watcher = FolderWatcher(conn)

        folder = tmp_path / "watched"
        folder.mkdir()

        conn.execute(
            "INSERT INTO folder_watches (path, label, scan_interval) VALUES (?, ?, ?)",
            (str(folder), "test", 300),
        )
        conn.commit()
        watch = _make_watch(str(folder))

        result = watcher.scan(watch)
        assert result.new_files == 0
        assert result.modified_files == 0
        assert result.deleted_files == 0

    def test_scan_detects_new_files(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        watcher = FolderWatcher(conn)

        folder = tmp_path / "watched"
        folder.mkdir()
        (folder / "hello.txt").write_text("hello world")
        (folder / "readme.md").write_text("# Readme")

        conn.execute(
            "INSERT INTO folder_watches (path, label, scan_interval) VALUES (?, ?, ?)",
            (str(folder), "test", 300),
        )
        conn.commit()
        watch = _make_watch(str(folder))

        result = watcher.scan(watch)
        assert result.new_files == 2
        assert result.modified_files == 0
        assert result.deleted_files == 0

    def test_scan_updates_file_count(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        watcher = FolderWatcher(conn)

        folder = tmp_path / "watched"
        folder.mkdir()
        (folder / "a.txt").write_text("a")
        (folder / "b.txt").write_text("b")

        conn.execute(
            "INSERT INTO folder_watches (path, label, scan_interval) VALUES (?, ?, ?)",
            (str(folder), "test", 300),
        )
        conn.commit()
        watch = _make_watch(str(folder))

        watcher.scan(watch)
        row = conn.execute("SELECT file_count FROM folder_watches WHERE id=1").fetchone()
        assert row[0] == 2

    def test_scan_nonexistent_folder(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        watcher = FolderWatcher(conn)

        watch = _make_watch(str(tmp_path / "nonexistent"))
        result = watcher.scan(watch)
        assert result.new_files == 0


class TestFolderWatcherChanges:
    def test_get_changes_detects_additions(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        watcher = FolderWatcher(conn)

        folder = tmp_path / "watched"
        folder.mkdir()
        (folder / "new.txt").write_text("new content")

        watch = _make_watch(str(folder))
        changes = watcher.get_changes(watch)

        assert len(changes) == 1
        assert changes[0].change_type == "added"
        assert "new.txt" in changes[0].path

    def test_get_changes_detects_deletions(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        watcher = FolderWatcher(conn)

        folder = tmp_path / "watched"
        folder.mkdir()

        # Add a file to the index that no longer exists on disk
        fake_path = str(folder / "deleted.txt").replace("\\", "/")
        conn.execute(
            "INSERT INTO content_index (source, content_type, summary, topics, indexed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fake_path, "text/plain", "old content", "txt", time.time()),
        )
        conn.commit()

        watch = _make_watch(str(folder))
        changes = watcher.get_changes(watch)

        deleted = [c for c in changes if c.change_type == "deleted"]
        assert len(deleted) == 1

    def test_get_changes_detects_modifications(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        watcher = FolderWatcher(conn)

        folder = tmp_path / "watched"
        folder.mkdir()
        file_path = folder / "existing.txt"
        file_path.write_text("original")

        # Index the file with an old timestamp
        normalized = str(file_path).replace("\\", "/")
        conn.execute(
            "INSERT INTO content_index (source, content_type, summary, topics, indexed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (normalized, "text/plain", "original", "txt", 1000.0),
        )
        conn.commit()

        watch = _make_watch(str(folder))
        changes = watcher.get_changes(watch)

        modified = [c for c in changes if c.change_type == "modified"]
        assert len(modified) == 1

    def test_scan_recursive(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        watcher = FolderWatcher(conn)

        folder = tmp_path / "watched"
        folder.mkdir()
        sub = folder / "sub"
        sub.mkdir()
        (folder / "top.txt").write_text("top")
        (sub / "deep.txt").write_text("deep")

        watch = _make_watch(str(folder))
        changes = watcher.get_changes(watch)

        assert len(changes) == 2
        paths = {c.path for c in changes}
        assert any("top.txt" in p for p in paths)
        assert any("deep.txt" in p for p in paths)
