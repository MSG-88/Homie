"""Mesh node registry — SQLite storage for known nodes in the mesh."""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from homie_core.utils import utc_now


@dataclass
class MeshNodeRecord:
    node_id: str
    node_name: str
    role: str
    mesh_id: str
    capability_score: float
    capabilities_json: str
    lan_ip: str
    tailnet_ip: str
    public_key_ed25519: str
    status: str
    last_seen_ts: str = ""
    paired_at: str = ""


class MeshNodeRegistry:
    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS mesh_nodes (
                node_id TEXT PRIMARY KEY,
                node_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'standalone',
                mesh_id TEXT,
                capability_score REAL DEFAULT 0,
                capabilities_json TEXT DEFAULT '{}',
                lan_ip TEXT DEFAULT '',
                tailnet_ip TEXT DEFAULT '',
                last_seen_ts TEXT DEFAULT '',
                paired_at TEXT DEFAULT '',
                public_key_ed25519 TEXT DEFAULT '',
                status TEXT DEFAULT 'offline'
            )
        """)
        self._conn.commit()

    def upsert(self, record: MeshNodeRecord) -> None:
        now = utc_now().isoformat()
        self._conn.execute("""
            INSERT INTO mesh_nodes (
                node_id, node_name, role, mesh_id, capability_score,
                capabilities_json, lan_ip, tailnet_ip, last_seen_ts,
                paired_at, public_key_ed25519, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                node_name=excluded.node_name, role=excluded.role,
                mesh_id=excluded.mesh_id, capability_score=excluded.capability_score,
                capabilities_json=excluded.capabilities_json, lan_ip=excluded.lan_ip,
                tailnet_ip=excluded.tailnet_ip, last_seen_ts=?,
                public_key_ed25519=excluded.public_key_ed25519, status=excluded.status
        """, (
            record.node_id, record.node_name, record.role, record.mesh_id,
            record.capability_score, record.capabilities_json,
            record.lan_ip, record.tailnet_ip,
            record.last_seen_ts or now, record.paired_at,
            record.public_key_ed25519, record.status, now,
        ))
        self._conn.commit()

    def get(self, node_id: str) -> Optional[MeshNodeRecord]:
        row = self._conn.execute(
            "SELECT * FROM mesh_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_all(self) -> list[MeshNodeRecord]:
        rows = self._conn.execute(
            "SELECT * FROM mesh_nodes ORDER BY capability_score DESC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def remove(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM mesh_nodes WHERE node_id = ?", (node_id,))
        self._conn.commit()

    def update_status(self, node_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE mesh_nodes SET status = ?, last_seen_ts = ? WHERE node_id = ?",
            (status, utc_now().isoformat(), node_id),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MeshNodeRecord:
        return MeshNodeRecord(
            node_id=row["node_id"],
            node_name=row["node_name"],
            role=row["role"],
            mesh_id=row["mesh_id"] or "",
            capability_score=row["capability_score"],
            capabilities_json=row["capabilities_json"],
            lan_ip=row["lan_ip"],
            tailnet_ip=row["tailnet_ip"],
            public_key_ed25519=row["public_key_ed25519"],
            status=row["status"],
            last_seen_ts=row["last_seen_ts"],
            paired_at=row["paired_at"],
        )
