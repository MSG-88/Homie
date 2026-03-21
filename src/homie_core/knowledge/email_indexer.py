"""EmailIndexer — indexes synced emails into the knowledge graph.

Reads from a SQLite cache table (populated by the email sync pipeline) and
creates Person + Document entities plus "authored" relationships.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from homie_core.knowledge.graph import KnowledgeGraph
from homie_core.knowledge.models import Entity, Relationship


class EmailIndexer:
    """Indexes emails from the sync cache into the knowledge graph."""

    def __init__(
        self,
        cache_db_path: str | Path,
        graph: Optional[KnowledgeGraph] = None,
    ):
        self._cache_path = str(cache_db_path)
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_recent(self, days: int = 30) -> dict:
        """Index emails from the last *days* days into the knowledge graph.

        Returns stats dict:
            {"indexed": int, "entities_created": int, "relationships_created": int}
        """
        stats: dict[str, int] = {
            "indexed": 0,
            "entities_created": 0,
            "relationships_created": 0,
        }

        if self._graph is None:
            return stats

        try:
            db = sqlite3.connect(self._cache_path)
            db.row_factory = sqlite3.Row
        except Exception:
            return stats

        try:
            cursor = db.execute(
                "SELECT * FROM emails WHERE date >= datetime('now', ?) ORDER BY date DESC LIMIT 500",
                (f"-{days} days",),
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            # Table doesn't exist — email not synced yet
            db.close()
            return stats

        for row in rows:
            self._index_email(dict(row), stats)

        db.close()
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_email(self, email: dict, stats: dict) -> None:
        """Extract entities from a single email and add to graph."""
        sender: str = email.get("sender", email.get("from_address", ""))
        subject: str = email.get("subject", "")

        sender_entity_id: Optional[str] = None

        if sender:
            # Parse display name from "Name <email>" format
            display_name = sender.split("<")[0].strip().strip('"')
            # Fall back to the full sender string if parsing yields nothing
            if not display_name:
                display_name = sender

            person = Entity(
                name=display_name,
                entity_type="person",
                attributes={"email": sender},
                source="email_sync",
            )
            sender_entity_id = self._graph.merge_entity(person)
            stats["entities_created"] += 1

        if subject:
            doc = Entity(
                name=subject[:100],
                entity_type="document",
                attributes={"format": "email", "from": sender},
                source="email_sync",
            )
            doc_id = self._graph.merge_entity(doc)
            stats["entities_created"] += 1

            # Link sender -> authored -> document
            if sender_entity_id:
                self._graph.add_relationship(
                    Relationship(
                        subject_id=sender_entity_id,
                        relation="authored",
                        object_id=doc_id,
                        source="email_sync",
                    )
                )
                stats["relationships_created"] += 1

        stats["indexed"] += 1
