"""Entity resolver — detects and merges duplicate entities."""

import json
import logging
from typing import Optional

from ..graph.store import KnowledgeGraphStore

logger = logging.getLogger(__name__)


def _similarity(a: str, b: str) -> float:
    """Simple normalized string similarity."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Jaccard on character bigrams
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s) - 1))
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 1.0 if a == b else 0.0
    return len(ba & bb) / len(ba | bb)


class EntityResolver:
    """Detects and merges duplicate entities in the knowledge graph."""

    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        self._graph = graph_store
        self._threshold = fuzzy_threshold

    def resolve(self, name: str, entity_type: str) -> Optional[dict]:
        """Try to find an existing entity matching this name."""
        # Exact name match
        existing = self._graph.find_entity_by_name(name)
        if existing and existing["entity_type"] == entity_type:
            return existing

        # Case-insensitive match already handled by find_entity_by_name
        if existing:
            return existing

        # Fuzzy match against all entities of same type — only if DB is small
        if self._graph.entity_count() > 1000:
            return None  # skip fuzzy for large graphs

        # Simple fuzzy: check if similarity exceeds threshold
        if self._graph._conn is None:
            return None
        rows = self._graph._conn.execute(
            "SELECT * FROM kg_entities WHERE entity_type = ?", (entity_type,)
        ).fetchall()
        for row in rows:
            if _similarity(name, row["name"]) >= self._threshold:
                d = dict(row)
                d["aliases"] = json.loads(d["aliases"])
                d["properties"] = json.loads(d["properties"])
                return d

        return None

    def merge(self, keep_id: str, remove_id: str) -> str:
        """Merge two entities — keep one, repoint relationships from other."""
        keep = self._graph.get_entity(keep_id)
        remove = self._graph.get_entity(remove_id)
        if not keep or not remove:
            return keep_id

        # Merge aliases
        combined_aliases = list(set(keep.get("aliases", []) + remove.get("aliases", []) + [remove["name"]]))
        if self._graph._conn:
            self._graph._conn.execute(
                "UPDATE kg_entities SET aliases = ? WHERE id = ?",
                (json.dumps(combined_aliases), keep_id),
            )
            # Repoint relationships
            self._graph._conn.execute(
                "UPDATE kg_relationships SET subject_id = ? WHERE subject_id = ?",
                (keep_id, remove_id),
            )
            self._graph._conn.execute(
                "UPDATE kg_relationships SET object_id = ? WHERE object_id = ?",
                (keep_id, remove_id),
            )
            # Remove merged entity
            self._graph._conn.execute("DELETE FROM kg_entities WHERE id = ?", (remove_id,))
            self._graph._conn.commit()

        logger.info("Merged entity '%s' into '%s'", remove["name"], keep["name"])
        return keep_id
