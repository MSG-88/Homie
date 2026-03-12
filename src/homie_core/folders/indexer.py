"""Content indexer — indexes file content into the content_index table.

Extracts basic metadata (size, type, modification time) and for text files
reads the first 1000 characters as a summary.
"""
from __future__ import annotations

import mimetypes
import os
import sqlite3
import time

from homie_core.folders.models import IndexedFile

# Maximum characters to read for text file summaries
_MAX_SUMMARY_CHARS = 1000


class ContentIndexer:
    """Indexes file content into the content_index table."""

    def __init__(self, cache_conn: sqlite3.Connection):
        self._conn = cache_conn

    def index_file(self, file_path: str) -> IndexedFile | None:
        """Index a single file. Returns IndexedFile or None on error.

        For text files, reads the first 1000 chars as a summary.
        Uses mimetypes to detect content type from file extension.
        """
        try:
            stat = os.stat(file_path)
        except OSError:
            return None

        content_type = self._detect_content_type(file_path)
        summary = self._extract_summary(file_path, content_type)
        topics = self._extract_topics(file_path)
        now = time.time()

        # Normalize path for consistent storage
        normalized_path = file_path.replace("\\", "/")

        # Upsert into content_index
        existing = self._conn.execute(
            "SELECT id FROM content_index WHERE source=?",
            (normalized_path,),
        ).fetchone()

        if existing:
            self._conn.execute(
                "UPDATE content_index SET content_type=?, summary=?, topics=?, indexed_at=? WHERE id=?",
                (content_type, summary, ",".join(topics), now, existing[0]),
            )
        else:
            self._conn.execute(
                "INSERT INTO content_index (source, content_type, summary, topics, indexed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (normalized_path, content_type, summary, ",".join(topics), now),
            )
        self._conn.commit()

        return IndexedFile(
            source=normalized_path,
            content_type=content_type,
            summary=summary,
            topics=topics,
            indexed_at=now,
            size=stat.st_size,
            modified_at=stat.st_mtime,
        )

    def remove_file(self, file_path: str) -> None:
        """Remove a file from the content index."""
        normalized = file_path.replace("\\", "/")
        self._conn.execute(
            "DELETE FROM content_index WHERE source=?",
            (normalized,),
        )
        self._conn.commit()

    def search(self, query: str, folder: str | None = None, max_results: int = 10) -> list[IndexedFile]:
        """Search indexed files by matching query against source path and summary.

        Uses SQL LIKE for matching. Optionally filter by folder prefix.
        """
        like_pattern = f"%{query}%"
        params: list = []

        sql = "SELECT source, content_type, summary, topics, indexed_at FROM content_index WHERE "
        conditions = ["(source LIKE ? OR summary LIKE ?)"]
        params.extend([like_pattern, like_pattern])

        if folder:
            normalized_folder = folder.replace("\\", "/")
            if not normalized_folder.endswith("/"):
                normalized_folder += "/"
            conditions.append("source LIKE ?")
            params.append(normalized_folder + "%")

        sql += " AND ".join(conditions)
        sql += " ORDER BY indexed_at DESC LIMIT ?"
        params.append(max_results)

        rows = self._conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            source = row[0]
            topics_str = row[3] or ""
            topics = [t.strip() for t in topics_str.split(",") if t.strip()]

            # Try to get file stats for size and mtime
            size = 0
            mtime = 0.0
            try:
                stat = os.stat(source)
                size = stat.st_size
                mtime = stat.st_mtime
            except OSError:
                pass

            results.append(IndexedFile(
                source=source,
                content_type=row[1],
                summary=row[2],
                topics=topics,
                indexed_at=row[4],
                size=size,
                modified_at=mtime,
            ))
        return results

    def _detect_content_type(self, file_path: str) -> str:
        """Detect content type from file extension using mimetypes."""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    def _extract_summary(self, file_path: str, content_type: str) -> str | None:
        """Extract summary for text files (first 1000 chars)."""
        if not content_type.startswith("text/"):
            return None
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(_MAX_SUMMARY_CHARS)
        except OSError:
            return None

    def _extract_topics(self, file_path: str) -> list[str]:
        """Extract basic topics from file path (extension, parent dir name)."""
        topics = []
        _, ext = os.path.splitext(file_path)
        if ext:
            topics.append(ext.lstrip("."))
        parent = os.path.basename(os.path.dirname(file_path))
        if parent:
            topics.append(parent)
        return topics
