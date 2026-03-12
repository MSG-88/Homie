"""Folder watcher — scans watched folders and detects changes.

Uses os.scandir for efficient directory traversal.
"""
from __future__ import annotations

import os
import sqlite3
import time

from homie_core.folders.models import FileChange, FolderWatch, ScanResult


class FolderWatcher:
    """Scans watched folders and detects new/changed/deleted files."""

    def __init__(self, cache_conn: sqlite3.Connection):
        self._conn = cache_conn

    def scan(self, watch: FolderWatch) -> ScanResult:
        """Scan a watched folder and update content_index with changes.

        Returns a ScanResult with counts of new, modified, and deleted files.
        """
        changes = self.get_changes(watch)
        result = ScanResult()
        for change in changes:
            if change.change_type == "added":
                result.new_files += 1
            elif change.change_type == "modified":
                result.modified_files += 1
            elif change.change_type == "deleted":
                result.deleted_files += 1

        # Update watch metadata
        file_count = self._count_files(watch.path)
        self._conn.execute(
            "UPDATE folder_watches SET last_scanned=?, file_count=? WHERE id=?",
            (time.time(), file_count, watch.id),
        )
        self._conn.commit()
        return result

    def get_changes(self, watch: FolderWatch) -> list[FileChange]:
        """Detect new, modified, and deleted files in a watched folder.

        Compares current filesystem state against content_index records.
        """
        changes: list[FileChange] = []

        if not os.path.isdir(watch.path):
            return changes

        # Get currently indexed files for this folder
        indexed = self._get_indexed_files(watch.path)

        # Scan filesystem
        current_files: dict[str, tuple[int, float]] = {}
        try:
            self._scan_dir(watch.path, current_files)
        except OSError:
            return changes

        # Detect additions and modifications
        for file_path, (size, mtime) in current_files.items():
            if file_path not in indexed:
                changes.append(FileChange(
                    path=file_path,
                    change_type="added",
                    size=size,
                    modified_at=mtime,
                ))
            else:
                indexed_at = indexed[file_path]
                if mtime > indexed_at:
                    changes.append(FileChange(
                        path=file_path,
                        change_type="modified",
                        size=size,
                        modified_at=mtime,
                    ))

        # Detect deletions
        for file_path in indexed:
            if file_path not in current_files:
                changes.append(FileChange(
                    path=file_path,
                    change_type="deleted",
                    size=0,
                    modified_at=0.0,
                ))

        return changes

    def _scan_dir(self, path: str, result: dict[str, tuple[int, float]]) -> None:
        """Recursively scan a directory using os.scandir."""
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        try:
                            stat = entry.stat(follow_symlinks=False)
                            # Normalize to forward slashes for cross-platform consistency
                            normalized = entry.path.replace("\\", "/")
                            result[normalized] = (stat.st_size, stat.st_mtime)
                        except OSError:
                            pass
                    elif entry.is_dir(follow_symlinks=False):
                        self._scan_dir(entry.path, result)
        except OSError:
            pass

    def _get_indexed_files(self, folder_path: str) -> dict[str, float]:
        """Get indexed files for a folder. Returns {path: indexed_at}."""
        # Normalize the folder path for LIKE query
        prefix = folder_path.replace("\\", "/")
        if not prefix.endswith("/"):
            prefix += "/"
        rows = self._conn.execute(
            "SELECT source, indexed_at FROM content_index WHERE source LIKE ?",
            (prefix + "%",),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def _count_files(self, path: str) -> int:
        """Count files in a directory recursively."""
        count = 0
        if not os.path.isdir(path):
            return 0
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        count += 1
                    elif entry.is_dir(follow_symlinks=False):
                        count += self._count_files(entry.path)
        except OSError:
            pass
        return count
