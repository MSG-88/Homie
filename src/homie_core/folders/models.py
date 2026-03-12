"""Folder awareness data models.

All models are plain dataclasses with no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FolderWatch:
    """A watched folder entry from the folder_watches table."""
    id: int
    path: str
    label: str | None
    scan_interval: int
    last_scanned: float | None
    file_count: int
    enabled: bool


@dataclass
class IndexedFile:
    """An indexed file entry from the content_index table."""
    source: str  # file path
    content_type: str  # e.g. "text/plain", "application/pdf"
    summary: str | None
    topics: list[str]
    indexed_at: float
    size: int  # bytes
    modified_at: float


@dataclass
class FileChange:
    """A detected change in a watched folder."""
    path: str
    change_type: str  # "added", "modified", "deleted"
    size: int
    modified_at: float


@dataclass
class ScanResult:
    """Summary of a folder scan."""
    new_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
