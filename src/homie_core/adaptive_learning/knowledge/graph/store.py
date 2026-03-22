"""KnowledgeGraphStore — SQLite CRUD for entities and temporal relationships."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


class KnowledgeGraphStore:
    """Persistent knowledge graph with temporal versioning."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Create database and tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '[]',
                properties TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kge_name ON kg_entities(name COLLATE NOCASE);
            CREATE INDEX IF NOT EXISTS idx_kge_type ON kg_entities(entity_type);

            CREATE TABLE IF NOT EXISTS kg_relationships (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                object_id TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source TEXT NOT NULL DEFAULT '',
                valid_from REAL NOT NULL,
                valid_until REAL,
                properties TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                FOREIGN KEY (subject_id) REFERENCES kg_entities(id),
                FOREIGN KEY (object_id) REFERENCES kg_entities(id)
            );
            CREATE INDEX IF NOT EXISTS idx_kgr_subject ON kg_relationships(subject_id);
            CREATE INDEX IF NOT EXISTS idx_kgr_object ON kg_relationships(object_id);
            CREATE INDEX IF NOT EXISTS idx_kgr_relation ON kg_relationships(relation);
            CREATE INDEX IF NOT EXISTS idx_kgr_valid ON kg_relationships(valid_until);
        """)
        self._conn.commit()

    def add_entity(
        self,
        name: str,
        entity_type: str,
        aliases: Optional[list[str]] = None,
        properties: Optional[dict] = None,
    ) -> str:
        """Add an entity. Returns entity ID."""
        if self._conn is None:
            return ""
        eid = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kg_entities (id, name, entity_type, aliases, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (eid, name, entity_type, json.dumps(aliases or []), json.dumps(properties or {}), now, now),
            )
            self._conn.commit()
        return eid

    def get_entity(self, entity_id: str) -> Optional[dict]:
        """Get entity by ID."""
        if self._conn is None:
            return None
        row = self._conn.execute("SELECT * FROM kg_entities WHERE id = ?", (entity_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["aliases"] = json.loads(d["aliases"])
        d["properties"] = json.loads(d["properties"])
        return d

    def find_entity_by_name(self, name: str) -> Optional[dict]:
        """Find entity by name or alias (case-insensitive)."""
        if self._conn is None:
            return None
        # Try exact name match first
        row = self._conn.execute(
            "SELECT * FROM kg_entities WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row:
            d = dict(row)
            d["aliases"] = json.loads(d["aliases"])
            d["properties"] = json.loads(d["properties"])
            return d
        # Search aliases
        rows = self._conn.execute("SELECT * FROM kg_entities").fetchall()
        name_lower = name.lower()
        for row in rows:
            aliases = json.loads(row["aliases"])
            if any(a.lower() == name_lower for a in aliases):
                d = dict(row)
                d["aliases"] = json.loads(d["aliases"])
                d["properties"] = json.loads(d["properties"])
                return d
        return None

    def add_relationship(
        self,
        subject_id: str,
        relation: str,
        object_id: str,
        confidence: float = 0.5,
        source: str = "",
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        properties: Optional[dict] = None,
    ) -> str:
        """Add a relationship. Returns relationship ID."""
        if self._conn is None:
            return ""
        rid = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kg_relationships (id, subject_id, relation, object_id, confidence, source, valid_from, valid_until, properties, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, subject_id, relation, object_id, confidence, source, valid_from or now, valid_until, json.dumps(properties or {}), now),
            )
            self._conn.commit()
        return rid

    def get_relationships(self, subject_id: Optional[str] = None, object_id: Optional[str] = None) -> list[dict]:
        """Get relationships by subject and/or object."""
        if self._conn is None:
            return []
        clauses, params = [], []
        if subject_id:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if object_id:
            clauses.append("object_id = ?")
            params.append(object_id)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(f"SELECT * FROM kg_relationships WHERE {where}", params).fetchall()
        return [dict(r) for r in rows]

    def find_current_relationships(self, subject_id: str, relation: str) -> list[dict]:
        """Find current (non-superseded) relationships."""
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM kg_relationships WHERE subject_id = ? AND relation = ? AND valid_until IS NULL",
            (subject_id, relation),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_relationships_at_time(self, subject_id: str, relation: str, timestamp: float) -> list[dict]:
        """Find relationships valid at a specific timestamp."""
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM kg_relationships WHERE subject_id = ? AND relation = ? AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)",
            (subject_id, relation, timestamp, timestamp),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_relationship_valid_until(self, relationship_id: str, valid_until: float) -> None:
        """Set valid_until on a relationship (supersede it)."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE kg_relationships SET valid_until = ? WHERE id = ?",
                (valid_until, relationship_id),
            )
            self._conn.commit()

    def entity_count(self) -> int:
        """Count total entities."""
        if self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) as c FROM kg_entities").fetchone()
        return row["c"]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
