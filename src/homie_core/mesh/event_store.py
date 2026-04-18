"""Event store — SQLite append-only log for mesh sync events."""
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Optional
from homie_core.mesh.events import HomieEvent

class EventStore:
    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                event_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                vector_clock_json TEXT NOT NULL DEFAULT '{}',
                checksum TEXT NOT NULL DEFAULT '',
                synced_to_hub BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_category ON event_log(category, timestamp)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_sync ON event_log(synced_to_hub, event_id)")
        self._conn.commit()

    def append(self, event: HomieEvent) -> None:
        self._conn.execute("""
            INSERT OR IGNORE INTO event_log (event_id, node_id, timestamp, category, event_type,
                payload_json, vector_clock_json, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (event.event_id, event.node_id, event.timestamp, event.category, event.event_type,
              json.dumps(event.payload), json.dumps(event.vector_clock), event.checksum))
        self._conn.commit()

    def get(self, event_id: str) -> Optional[HomieEvent]:
        row = self._conn.execute("SELECT * FROM event_log WHERE event_id = ?", (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    def events_since(self, after_event_id: Optional[str], limit: int = 1000) -> list[HomieEvent]:
        if after_event_id is None:
            rows = self._conn.execute("SELECT * FROM event_log ORDER BY event_id ASC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM event_log WHERE event_id > ? ORDER BY event_id ASC LIMIT ?",
                                      (after_event_id, limit)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def events_by_category(self, category: str, limit: int = 1000) -> list[HomieEvent]:
        rows = self._conn.execute("SELECT * FROM event_log WHERE category = ? ORDER BY event_id ASC LIMIT ?",
                                  (category, limit)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def unsynced_events(self, limit: int = 1000) -> list[HomieEvent]:
        rows = self._conn.execute("SELECT * FROM event_log WHERE synced_to_hub = 0 ORDER BY event_id ASC LIMIT ?",
                                  (limit,)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def mark_synced(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        self._conn.execute(f"UPDATE event_log SET synced_to_hub = 1 WHERE event_id IN ({placeholders})", event_ids)
        self._conn.commit()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM event_log").fetchone()[0]

    def last_event_id(self) -> Optional[str]:
        row = self._conn.execute("SELECT event_id FROM event_log ORDER BY event_id DESC LIMIT 1").fetchone()
        return row[0] if row else None

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> HomieEvent:
        return HomieEvent(event_id=row["event_id"], node_id=row["node_id"], timestamp=row["timestamp"],
                          category=row["category"], event_type=row["event_type"],
                          payload=json.loads(row["payload_json"]),
                          vector_clock=json.loads(row["vector_clock_json"]), checksum=row["checksum"])
