import time
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore


class TestKnowledgeGraphStore:
    def test_add_and_get_entity(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        eid = store.add_entity("Python", "technology", aliases=["Python3", "CPython"])
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity["name"] == "Python"
        assert "Python3" in entity["aliases"]

    def test_add_and_get_relationship(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        e1 = store.add_entity("User", "person")
        e2 = store.add_entity("Google", "organization")
        rid = store.add_relationship(e1, "works_at", e2, confidence=0.9, source="conversation")
        rels = store.get_relationships(subject_id=e1)
        assert len(rels) == 1
        assert rels[0]["relation"] == "works_at"
        assert rels[0]["valid_until"] is None

    def test_find_current_relationships(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        e1 = store.add_entity("User", "person")
        e2 = store.add_entity("Google", "organization")
        e3 = store.add_entity("Anthropic", "organization")
        # Old superseded relationship
        store.add_relationship(e1, "works_at", e2, confidence=0.9, source="conv", valid_until=time.time() - 100)
        # Current relationship
        store.add_relationship(e1, "works_at", e3, confidence=0.95, source="conv")
        current = store.find_current_relationships(e1, "works_at")
        assert len(current) == 1
        assert current[0]["object_id"] == e3

    def test_find_entity_by_name(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        store.add_entity("Python", "technology", aliases=["Python3"])
        result = store.find_entity_by_name("Python")
        assert result is not None
        result2 = store.find_entity_by_name("Python3")
        assert result2 is not None  # found via alias

    def test_find_entity_by_name_case_insensitive(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        store.add_entity("Python", "technology")
        assert store.find_entity_by_name("python") is not None

    def test_update_relationship_valid_until(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        e1 = store.add_entity("A", "thing")
        e2 = store.add_entity("B", "thing")
        rid = store.add_relationship(e1, "uses", e2, confidence=0.8, source="test")
        now = time.time()
        store.update_relationship_valid_until(rid, now)
        rels = store.get_relationships(subject_id=e1)
        assert rels[0]["valid_until"] is not None

    def test_find_relationships_at_time(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        e1 = store.add_entity("User", "person")
        e2 = store.add_entity("Google", "organization")
        past = time.time() - 1000
        store.add_relationship(e1, "works_at", e2, confidence=0.9, source="test", valid_from=past - 500, valid_until=past - 100)
        # Query at a time when the relationship was valid
        results = store.find_relationships_at_time(e1, "works_at", past - 300)
        assert len(results) == 1

    def test_entity_count(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        store.add_entity("A", "thing")
        store.add_entity("B", "thing")
        assert store.entity_count() == 2

    def test_close(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        store.close()
        store.close()  # double close safe
