"""Memory sync engine — merge strategies for LAN device sync."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SyncEntry:
    """A single syncable entry."""
    id: str
    device_id: str
    timestamp: int  # Unix timestamp or monotonic counter
    data: Optional[dict] = None
    content_hash: str = ""
    tombstone: bool = False
    lamport: int = 0


@dataclass
class SyncResult:
    """Result of a merge operation."""
    merged: list[SyncEntry] = field(default_factory=list)
    new_entries: int = 0
    deleted: int = 0


class SyncEngine:
    """Merges data between devices using per-type strategies."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._sync_versions: dict[str, int] = {}

    def get_sync_version(self, data_type: str) -> int:
        return self._sync_versions.get(data_type, 0)

    def increment_sync_version(self, data_type: str) -> int:
        v = self._sync_versions.get(data_type, 0) + 1
        self._sync_versions[data_type] = v
        return v

    def merge_conversations(
        self, local: list[SyncEntry], remote: list[SyncEntry]
    ) -> SyncResult:
        """Append-only merge — no conflicts possible. Dedup by ID."""
        seen_ids: set[str] = set()
        merged: list[SyncEntry] = []
        new_count = 0

        local_ids = {e.id for e in local}

        for entry in local:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                merged.append(entry)

        for entry in remote:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                merged.append(entry)
                if entry.id not in local_ids:
                    new_count += 1

        merged.sort(key=lambda e: e.timestamp)
        self.increment_sync_version("conversation")
        return SyncResult(merged=merged, new_entries=new_count)

    def merge_episodic(
        self, local: list[SyncEntry], remote: list[SyncEntry]
    ) -> SyncResult:
        """Append-only merge — dedup by content_hash."""
        seen_hashes: set[str] = set()
        merged: list[SyncEntry] = []
        new_count = 0

        local_hashes = set()
        for entry in local:
            h = entry.content_hash or entry.id
            if h not in seen_hashes:
                seen_hashes.add(h)
                local_hashes.add(h)
                merged.append(entry)

        for entry in remote:
            h = entry.content_hash or entry.id
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(entry)
                if h not in local_hashes:
                    new_count += 1

        merged.sort(key=lambda e: e.timestamp)
        self.increment_sync_version("episodic")
        return SyncResult(merged=merged, new_entries=new_count)

    def merge_semantic(
        self, local: list[SyncEntry], remote: list[SyncEntry]
    ) -> SyncResult:
        """Additive merge — deletions via tombstone with Lamport timestamp."""
        by_id: dict[str, SyncEntry] = {}
        for entry in local:
            by_id[entry.id] = entry
        for entry in remote:
            existing = by_id.get(entry.id)
            if existing is None:
                by_id[entry.id] = entry
            elif entry.lamport > existing.lamport:
                by_id[entry.id] = entry
            elif entry.lamport == existing.lamport and entry.device_id > existing.device_id:
                by_id[entry.id] = entry

        deleted = 0
        merged: list[SyncEntry] = []
        for entry in by_id.values():
            if entry.tombstone:
                deleted += 1
            else:
                merged.append(entry)

        new_count = len(merged) - len([e for e in local if not e.tombstone])
        self.increment_sync_version("semantic")
        return SyncResult(merged=merged, new_entries=max(0, new_count), deleted=deleted)

    def merge_settings(
        self, local: dict[str, dict], remote: dict[str, dict]
    ) -> dict[str, dict]:
        """Last-write-wins with Lamport counters. Tie-break by device_id (lexicographic)."""
        result = dict(local)
        for key, remote_val in remote.items():
            if key not in result:
                result[key] = remote_val
            else:
                local_val = result[key]
                local_lamport = local_val.get("lamport", 0)
                remote_lamport = remote_val.get("lamport", 0)
                if remote_lamport > local_lamport:
                    result[key] = remote_val
                elif remote_lamport == local_lamport:
                    if remote_val.get("device_id", "") > local_val.get("device_id", ""):
                        result[key] = remote_val
        self.increment_sync_version("settings")
        return result
