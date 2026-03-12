"""Folder awareness — watch directories, index content, and provide search.

FolderService is the main facade used by the daemon and CLI.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from homie_core.folders.models import FolderWatch, IndexedFile
from homie_core.folders.watcher import FolderWatcher
from homie_core.folders.indexer import ContentIndexer


class FolderService:
    """High-level facade for folder awareness operations.

    Used by the daemon for sync callbacks and by tools for queries.
    """

    def __init__(self, cache_conn: sqlite3.Connection, working_memory=None):
        self._conn = cache_conn
        self._working_memory = working_memory
        self._watcher = FolderWatcher(cache_conn)
        self._indexer = ContentIndexer(cache_conn)

    def add_watch(self, path: str, label: str | None = None, scan_interval: int = 300) -> dict:
        """Add a folder to watch. Returns watch info dict."""
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return {"error": f"Directory not found: {abs_path}"}

        try:
            self._conn.execute(
                "INSERT INTO folder_watches (path, label, scan_interval) VALUES (?, ?, ?)",
                (abs_path, label, scan_interval),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            return {"error": f"Already watching: {abs_path}"}

        row = self._conn.execute(
            "SELECT id, path, label, scan_interval, last_scanned, file_count, enabled "
            "FROM folder_watches WHERE path=?",
            (abs_path,),
        ).fetchone()

        return self._row_to_dict(row)

    def remove_watch(self, path: str) -> bool:
        """Remove a folder watch. Returns True if removed."""
        abs_path = os.path.abspath(path)
        cursor = self._conn.execute(
            "DELETE FROM folder_watches WHERE path=?",
            (abs_path,),
        )
        self._conn.commit()

        if cursor.rowcount > 0:
            # Clean up indexed files for this folder
            normalized = abs_path.replace("\\", "/")
            if not normalized.endswith("/"):
                normalized += "/"
            self._conn.execute(
                "DELETE FROM content_index WHERE source LIKE ?",
                (normalized + "%",),
            )
            self._conn.commit()
            return True
        return False

    def list_watches(self) -> list[dict]:
        """List all folder watches."""
        rows = self._conn.execute(
            "SELECT id, path, label, scan_interval, last_scanned, file_count, enabled "
            "FROM folder_watches ORDER BY id",
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def scan_tick(self) -> str:
        """Called by SyncManager on each tick. Scans all due folders.

        Returns a string summary like "3 new, 1 modified in Documents".
        """
        now = time.time()
        rows = self._conn.execute(
            "SELECT id, path, label, scan_interval, last_scanned, file_count, enabled "
            "FROM folder_watches WHERE enabled=1",
        ).fetchall()

        parts = []
        for row in rows:
            watch = self._row_to_watch(row)
            # Check if scan is due
            if watch.last_scanned and (now - watch.last_scanned) < watch.scan_interval:
                continue

            result = self._watcher.scan(watch)
            changes = self._watcher.get_changes(watch)

            # Index new and modified files
            for change in changes:
                if change.change_type in ("added", "modified"):
                    self._indexer.index_file(change.path)
                elif change.change_type == "deleted":
                    self._indexer.remove_file(change.path)

            label = watch.label or os.path.basename(watch.path)
            change_parts = []
            if result.new_files:
                change_parts.append(f"{result.new_files} new")
            if result.modified_files:
                change_parts.append(f"{result.modified_files} modified")
            if result.deleted_files:
                change_parts.append(f"{result.deleted_files} deleted")
            if change_parts:
                parts.append(f"{', '.join(change_parts)} in {label}")

        return "; ".join(parts) if parts else "No changes"

    def search(self, query: str, folder: str | None = None, max_results: int = 10) -> list[IndexedFile]:
        """Search indexed files."""
        return self._indexer.search(query, folder=folder, max_results=max_results)

    def get_summary(self, folder: str | None = None) -> dict:
        """Get folder overview: watches, file counts, types."""
        if folder:
            abs_path = os.path.abspath(folder)
            row = self._conn.execute(
                "SELECT id, path, label, scan_interval, last_scanned, file_count, enabled "
                "FROM folder_watches WHERE path=?",
                (abs_path,),
            ).fetchone()
            if not row:
                return {"error": f"Not watching: {abs_path}"}

            watch = self._row_to_watch(row)
            normalized = abs_path.replace("\\", "/")
            if not normalized.endswith("/"):
                normalized += "/"

            type_counts = self._conn.execute(
                "SELECT content_type, COUNT(*) FROM content_index WHERE source LIKE ? GROUP BY content_type",
                (normalized + "%",),
            ).fetchall()

            return {
                "path": watch.path,
                "label": watch.label,
                "file_count": watch.file_count,
                "enabled": watch.enabled,
                "last_scanned": watch.last_scanned,
                "content_types": {row[0]: row[1] for row in type_counts},
            }

        # Summary of all watches
        watches = self.list_watches()
        total_files = sum(w["file_count"] for w in watches)
        total_indexed = self._conn.execute(
            "SELECT COUNT(*) FROM content_index",
        ).fetchone()[0]

        return {
            "watch_count": len(watches),
            "total_files": total_files,
            "total_indexed": total_indexed,
            "watches": watches,
        }

    def _row_to_watch(self, row: tuple) -> FolderWatch:
        """Convert a DB row to a FolderWatch model."""
        return FolderWatch(
            id=row[0],
            path=row[1],
            label=row[2],
            scan_interval=row[3],
            last_scanned=row[4],
            file_count=row[5],
            enabled=bool(row[6]),
        )

    def _row_to_dict(self, row: tuple) -> dict:
        """Convert a DB row to a dict."""
        return {
            "id": row[0],
            "path": row[1],
            "label": row[2],
            "scan_interval": row[3],
            "last_scanned": row[4],
            "file_count": row[5],
            "enabled": bool(row[6]),
        }
