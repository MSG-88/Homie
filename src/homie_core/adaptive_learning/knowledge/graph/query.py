"""GraphQuery — query current/historical facts and traverse relationships."""

from typing import Optional
from .store import KnowledgeGraphStore


class GraphQuery:
    """High-level query interface for the knowledge graph."""

    def __init__(self, store: KnowledgeGraphStore) -> None:
        self._store = store

    def get_entity_relationships(self, entity_id: str, current_only: bool = True) -> list[dict]:
        """Get all relationships for an entity (as subject or object)."""
        as_subject = self._store.get_relationships(subject_id=entity_id)
        as_object = self._store.get_relationships(object_id=entity_id)
        all_rels = as_subject + as_object
        if current_only:
            all_rels = [r for r in all_rels if r.get("valid_until") is None]
        return all_rels

    def get_related_entities(self, entity_id: str, relation: Optional[str] = None) -> list[dict]:
        """Get entities related to a given entity via current relationships."""
        rels = self._store.find_current_relationships(entity_id, relation) if relation else [
            r for r in self._store.get_relationships(subject_id=entity_id) if r.get("valid_until") is None
        ]
        entities = []
        for r in rels:
            obj = self._store.get_entity(r["object_id"])
            if obj:
                entities.append(obj)
        return entities

    def traverse(self, start_entity_id: str, max_hops: int = 2) -> list[dict]:
        """Traverse the graph from a starting entity up to max_hops."""
        visited = set()
        result = []
        queue = [(start_entity_id, 0)]

        while queue:
            eid, depth = queue.pop(0)
            if eid in visited or depth > max_hops:
                continue
            visited.add(eid)

            entity = self._store.get_entity(eid)
            if entity and eid != start_entity_id:
                result.append(entity)

            if depth < max_hops:
                rels = [r for r in self._store.get_relationships(subject_id=eid) if r.get("valid_until") is None]
                for r in rels:
                    if r["object_id"] not in visited:
                        queue.append((r["object_id"], depth + 1))

        return result

    def get_entity_summary(self, entity_id: str) -> dict:
        """Get a summary of an entity and its relationships."""
        entity = self._store.get_entity(entity_id)
        if entity is None:
            return {}
        rels = self.get_entity_relationships(entity_id, current_only=True)
        return {**entity, "relationships": rels}

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> list[dict]:
        """Search entities by name (case-insensitive)."""
        result = self._store.find_entity_by_name(query)
        if result:
            if entity_type and result["entity_type"] != entity_type:
                return []
            return [result]
        return []
