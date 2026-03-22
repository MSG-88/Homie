import time
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.graph.query import GraphQuery


class TestGraphQuery:
    def _setup_graph(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        user = store.add_entity("User", "person")
        homie = store.add_entity("Homie", "project")
        python = store.add_entity("Python", "technology")
        chromadb = store.add_entity("ChromaDB", "technology")
        store.add_relationship(user, "works_on", homie, confidence=0.95, source="conversation")
        store.add_relationship(homie, "uses", python, confidence=0.9, source="code_scan")
        store.add_relationship(homie, "uses", chromadb, confidence=0.85, source="code_scan")
        return store, {"user": user, "homie": homie, "python": python, "chromadb": chromadb}

    def test_get_entity_relationships(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        rels = query.get_entity_relationships(ids["homie"])
        assert len(rels) >= 2  # uses Python, uses ChromaDB

    def test_get_related_entities(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        related = query.get_related_entities(ids["homie"], relation="uses")
        names = [e["name"] for e in related]
        assert "Python" in names
        assert "ChromaDB" in names

    def test_traverse_one_hop(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        # User works_on Homie -> Homie uses Python
        reachable = query.traverse(ids["user"], max_hops=2)
        entity_ids = [e["id"] for e in reachable]
        assert ids["python"] in entity_ids

    def test_get_entity_summary(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        summary = query.get_entity_summary(ids["homie"])
        assert summary["name"] == "Homie"
        assert len(summary["relationships"]) >= 2

    def test_search_entities(self, tmp_path):
        store, ids = self._setup_graph(tmp_path)
        query = GraphQuery(store=store)
        results = query.search_entities("python")
        assert len(results) >= 1
        assert results[0]["name"] == "Python"
