"""Tests for ContentIndexer — indexing, search, and removal."""
from __future__ import annotations

import os
import time
from pathlib import Path

from homie_core.folders.indexer import ContentIndexer
from homie_core.vault.schema import create_cache_db


class TestContentIndexerIndex:
    def test_index_text_file(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        file_path = tmp_path / "hello.txt"
        file_path.write_text("Hello world, this is a test file.")

        result = indexer.index_file(str(file_path))
        assert result is not None
        assert result.content_type == "text/plain"
        assert "Hello world" in result.summary
        assert result.size > 0

    def test_index_creates_db_entry(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        file_path = tmp_path / "data.txt"
        file_path.write_text("some data")

        indexer.index_file(str(file_path))
        normalized = str(file_path).replace("\\", "/")
        row = conn.execute(
            "SELECT source, content_type FROM content_index WHERE source=?",
            (normalized,),
        ).fetchone()
        assert row is not None
        assert row[1] == "text/plain"

    def test_index_nonexistent_file_returns_none(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        result = indexer.index_file(str(tmp_path / "nope.txt"))
        assert result is None

    def test_index_binary_file_no_summary(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        file_path = tmp_path / "image.png"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = indexer.index_file(str(file_path))
        assert result is not None
        assert result.content_type == "image/png"
        assert result.summary is None

    def test_index_updates_existing(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        file_path = tmp_path / "update.txt"
        file_path.write_text("version 1")
        indexer.index_file(str(file_path))

        file_path.write_text("version 2")
        result = indexer.index_file(str(file_path))
        assert "version 2" in result.summary

        # Should still be only one entry
        normalized = str(file_path).replace("\\", "/")
        count = conn.execute(
            "SELECT COUNT(*) FROM content_index WHERE source=?",
            (normalized,),
        ).fetchone()[0]
        assert count == 1

    def test_index_truncates_long_text(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        file_path = tmp_path / "long.txt"
        file_path.write_text("x" * 5000)

        result = indexer.index_file(str(file_path))
        assert result is not None
        assert len(result.summary) == 1000


class TestContentIndexerRemove:
    def test_remove_file(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        file_path = tmp_path / "gone.txt"
        file_path.write_text("remove me")
        indexer.index_file(str(file_path))

        indexer.remove_file(str(file_path))
        normalized = str(file_path).replace("\\", "/")
        row = conn.execute(
            "SELECT COUNT(*) FROM content_index WHERE source=?",
            (normalized,),
        ).fetchone()
        assert row[0] == 0

    def test_remove_nonexistent_is_noop(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)
        # Should not raise
        indexer.remove_file(str(tmp_path / "nope.txt"))


class TestContentIndexerSearch:
    def test_search_by_filename(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        file1 = tmp_path / "report.txt"
        file1.write_text("quarterly earnings report")
        file2 = tmp_path / "notes.txt"
        file2.write_text("meeting notes")

        indexer.index_file(str(file1))
        indexer.index_file(str(file2))

        results = indexer.search("report")
        assert len(results) >= 1
        assert any("report" in r.source for r in results)

    def test_search_by_summary_content(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        file1 = tmp_path / "doc.txt"
        file1.write_text("The budget meeting is scheduled for Tuesday.")
        indexer.index_file(str(file1))

        results = indexer.search("budget")
        assert len(results) == 1
        assert "budget" in results[0].summary

    def test_search_with_folder_filter(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        folder_a = tmp_path / "a"
        folder_b = tmp_path / "b"
        folder_a.mkdir()
        folder_b.mkdir()

        (folder_a / "match.txt").write_text("target content")
        (folder_b / "match.txt").write_text("target content")

        indexer.index_file(str(folder_a / "match.txt"))
        indexer.index_file(str(folder_b / "match.txt"))

        results = indexer.search("target", folder=str(folder_a))
        assert len(results) == 1
        assert str(folder_a).replace("\\", "/") in results[0].source

    def test_search_respects_max_results(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexer = ContentIndexer(conn)

        for i in range(5):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"common keyword {i}")
            indexer.index_file(str(f))

        results = indexer.search("common", max_results=2)
        assert len(results) == 2
