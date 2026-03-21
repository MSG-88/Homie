"""SQLite-backed triple store for the Homie Knowledge Graph.

Entities are nodes; Relationships are directed edges.
Provides BFS traversal, natural-language context generation, and
maintenance operations (decay + prune).
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from homie_core.knowledge.models import Entity, Relationship


class KnowledgeGraph:
    """SQLite triple store with entity and relationship management.

    Thread-safety: uses a single sqlite3 connection with WAL mode.
    Not safe for concurrent multi-process access to the same file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.row_factory = sqlite3.Row
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        cur = self._db
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                attributes  TEXT NOT NULL DEFAULT '{}',
                confidence  REAL NOT NULL DEFAULT 1.0,
                source      TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT '',
                updated_at  TEXT NOT NULL DEFAULT '',
                last_accessed TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name
                ON entities (name COLLATE NOCASE);

            CREATE INDEX IF NOT EXISTS idx_entities_type
                ON entities (entity_type);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_type
                ON entities (name COLLATE NOCASE, entity_type);

            CREATE TABLE IF NOT EXISTS relationships (
                id              TEXT PRIMARY KEY,
                subject_id      TEXT NOT NULL,
                relation        TEXT NOT NULL,
                object_id       TEXT NOT NULL,
                confidence      REAL NOT NULL DEFAULT 1.0,
                source          TEXT NOT NULL DEFAULT '',
                source_chunk_id TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL DEFAULT '',
                updated_at      TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (subject_id) REFERENCES entities(id),
                FOREIGN KEY (object_id)  REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_rel_subject
                ON relationships (subject_id);

            CREATE INDEX IF NOT EXISTS idx_rel_object
                ON relationships (object_id);

            CREATE INDEX IF NOT EXISTS idx_rel_relation
                ON relationships (relation);
        """)
        self._db.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            attributes=json.loads(row["attributes"]),
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
        )

    @staticmethod
    def _row_to_relationship(row: sqlite3.Row) -> Relationship:
        return Relationship(
            id=row["id"],
            subject_id=row["subject_id"],
            relation=row["relation"],
            object_id=row["object_id"],
            confidence=row["confidence"],
            source=row["source"],
            source_chunk_id=row["source_chunk_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> str:
        """Insert entity and return its id."""
        now = self._now()
        created_at = entity.created_at or now
        updated_at = entity.updated_at or now
        self._db.execute(
            """
            INSERT INTO entities
                (id, name, entity_type, attributes, confidence, source,
                 created_at, updated_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.id,
                entity.name,
                entity.entity_type,
                json.dumps(entity.attributes),
                entity.confidence,
                entity.source,
                created_at,
                updated_at,
                entity.last_accessed or now,
            ),
        )
        self._db.commit()
        return entity.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Fetch entity by id or None."""
        row = self._db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[Entity]:
        """Find entities by optional name (case-insensitive) and/or type."""
        query = "SELECT * FROM entities WHERE 1=1"
        params: list = []
        if name is not None:
            query += " AND name LIKE ? COLLATE NOCASE"
            params.append(f"%{name}%")
        if entity_type is not None:
            query += " AND entity_type = ?"
            params.append(entity_type)
        query += " LIMIT ?"
        params.append(limit)
        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def merge_entity(self, entity: Entity) -> str:
        """Upsert: if entity with same name+type exists, update it; otherwise insert.

        Returns the id of the canonical entity.
        """
        # Check for existing entity with same name (case-insensitive) + type
        row = self._db.execute(
            "SELECT id FROM entities WHERE name = ? COLLATE NOCASE AND entity_type = ?",
            (entity.name, entity.entity_type),
        ).fetchone()

        if row:
            existing_id = row["id"]
            now = self._now()
            self._db.execute(
                """
                UPDATE entities
                SET confidence = ?, source = ?, attributes = ?,
                    updated_at = ?, last_accessed = ?
                WHERE id = ?
                """,
                (
                    entity.confidence,
                    entity.source,
                    json.dumps(entity.attributes),
                    now,
                    now,
                    existing_id,
                ),
            )
            self._db.commit()
            return existing_id
        else:
            return self.add_entity(entity)

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def add_relationship(self, rel: Relationship) -> str:
        """Insert a relationship and return its id."""
        now = self._now()
        self._db.execute(
            """
            INSERT INTO relationships
                (id, subject_id, relation, object_id, confidence, source,
                 source_chunk_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rel.id,
                rel.subject_id,
                rel.relation,
                rel.object_id,
                rel.confidence,
                rel.source,
                rel.source_chunk_id,
                rel.created_at or now,
                rel.updated_at or now,
            ),
        )
        self._db.commit()
        return rel.id

    def get_relationships(
        self,
        entity_id: str,
        relation: Optional[str] = None,
        direction: str = "both",
    ) -> list[Relationship]:
        """Get relationships for an entity.

        direction: "outgoing" (entity is subject), "incoming" (entity is object),
                   or "both"
        """
        clauses: list[str] = []
        params: list = []

        if direction == "outgoing":
            clauses.append("subject_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            clauses.append("object_id = ?")
            params.append(entity_id)
        else:  # both
            clauses.append("(subject_id = ? OR object_id = ?)")
            params.extend([entity_id, entity_id])

        if relation is not None:
            clauses.append("relation = ?")
            params.append(relation)

        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._db.execute(
            f"SELECT * FROM relationships WHERE {where}", params
        ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def neighbors(self, entity_id: str, max_hops: int = 2) -> list[Entity]:
        """BFS to find entities reachable within max_hops of the given entity.

        Does not include the starting entity itself.
        """
        if self.get_entity(entity_id) is None:
            return []

        visited: set[str] = {entity_id}
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
        result: list[Entity] = []

        while queue:
            current_id, hop = queue.popleft()
            if hop >= max_hops:
                continue

            # Get all relationships (both directions)
            rels = self.get_relationships(current_id, direction="both")
            for rel in rels:
                neighbor_id = rel.object_id if rel.subject_id == current_id else rel.subject_id
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    entity = self.get_entity(neighbor_id)
                    if entity:
                        result.append(entity)
                        queue.append((neighbor_id, hop + 1))

        return result

    # ------------------------------------------------------------------
    # Query support
    # ------------------------------------------------------------------

    def entities_mentioned_in(self, text: str) -> list[Entity]:
        """Find entities whose names appear in text (case-insensitive)."""
        if not text:
            return []
        text_lower = text.lower()
        # Fetch all entities and check substring match
        rows = self._db.execute("SELECT * FROM entities").fetchall()
        return [
            self._row_to_entity(r)
            for r in rows
            if r["name"].lower() in text_lower
        ]

    def context_for_entity(self, entity_id: str) -> str:
        """Generate a natural-language summary of the entity and its relationships.

        Example: "Alice (person) works_on Homie, authored README.md"
        """
        entity = self.get_entity(entity_id)
        if entity is None:
            return ""

        parts: list[str] = []
        rels = self.get_relationships(entity_id, direction="both")

        for rel in rels:
            if rel.subject_id == entity_id:
                # entity -> relation -> other
                other = self.get_entity(rel.object_id)
                if other:
                    parts.append(f"{rel.relation} {other.name}")
            else:
                # other -> relation -> entity (passive)
                other = self.get_entity(rel.subject_id)
                if other:
                    parts.append(f"{rel.relation}_by {other.name}")

        summary = f"{entity.name} ({entity.entity_type})"
        if parts:
            summary += " " + ", ".join(parts)
        return summary

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay_scores(self, half_life_days: int = 30) -> None:
        """Reduce confidence of entities/relationships based on age.

        Uses exponential decay: confidence *= 0.5 ^ (days_old / half_life_days)
        Timestamps in updated_at drive the age calculation. Entities with
        no updated_at are treated as "now" (no decay).
        """
        now = datetime.now(timezone.utc)
        half_life_seconds = half_life_days * 86400

        # Decay entities
        rows = self._db.execute("SELECT id, confidence, updated_at FROM entities").fetchall()
        for row in rows:
            if not row["updated_at"]:
                continue
            try:
                updated = datetime.fromisoformat(row["updated_at"])
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                age_seconds = (now - updated).total_seconds()
                if age_seconds <= 0:
                    continue
                decay = math.pow(0.5, age_seconds / half_life_seconds)
                new_confidence = row["confidence"] * decay
                self._db.execute(
                    "UPDATE entities SET confidence = ? WHERE id = ?",
                    (new_confidence, row["id"]),
                )
            except (ValueError, TypeError):
                continue

        # Decay relationships
        rows = self._db.execute("SELECT id, confidence, updated_at FROM relationships").fetchall()
        for row in rows:
            if not row["updated_at"]:
                continue
            try:
                updated = datetime.fromisoformat(row["updated_at"])
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                age_seconds = (now - updated).total_seconds()
                if age_seconds <= 0:
                    continue
                decay = math.pow(0.5, age_seconds / half_life_seconds)
                new_confidence = row["confidence"] * decay
                self._db.execute(
                    "UPDATE relationships SET confidence = ? WHERE id = ?",
                    (new_confidence, row["id"]),
                )
            except (ValueError, TypeError):
                continue

        self._db.commit()

    def prune(self, min_confidence: float = 0.1) -> None:
        """Delete entities and relationships with confidence below min_confidence."""
        self._db.execute(
            "DELETE FROM relationships WHERE confidence < ?", (min_confidence,)
        )
        self._db.execute(
            "DELETE FROM entities WHERE confidence < ?", (min_confidence,)
        )
        self._db.commit()

    def stats(self) -> dict:
        """Return counts of entities by type and relationships by type."""
        entity_count = self._db.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]

        relationship_count = self._db.execute(
            "SELECT COUNT(*) FROM relationships"
        ).fetchone()[0]

        type_rows = self._db.execute(
            "SELECT entity_type, COUNT(*) as cnt FROM entities GROUP BY entity_type"
        ).fetchall()
        entities_by_type = {r["entity_type"]: r["cnt"] for r in type_rows}

        rel_rows = self._db.execute(
            "SELECT relation, COUNT(*) as cnt FROM relationships GROUP BY relation"
        ).fetchall()
        relationships_by_type = {r["relation"]: r["cnt"] for r in rel_rows}

        return {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "entities_by_type": entities_by_type,
            "relationships_by_type": relationships_by_type,
        }
