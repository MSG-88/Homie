# tests/unit/knowledge_evolution/test_entity_resolver.py
import pytest
from homie_core.adaptive_learning.knowledge.graph.store import KnowledgeGraphStore
from homie_core.adaptive_learning.knowledge.reasoning.entity_resolver import EntityResolver


class TestEntityResolver:
    def _setup(self, tmp_path):
        store = KnowledgeGraphStore(db_path=tmp_path / "kg.db")
        store.initialize()
        return store

    def test_exact_name_match(self, tmp_path):
        store = self._setup(tmp_path)
        e1 = store.add_entity("Python", "technology")
        resolver = EntityResolver(graph_store=store)
        match = resolver.resolve("Python", "technology")
        assert match is not None
        assert match["id"] == e1

    def test_alias_match(self, tmp_path):
        store = self._setup(tmp_path)
        e1 = store.add_entity("Python", "technology", aliases=["Python3", "CPython"])
        resolver = EntityResolver(graph_store=store)
        match = resolver.resolve("Python3", "technology")
        assert match is not None
        assert match["id"] == e1

    def test_no_match_returns_none(self, tmp_path):
        store = self._setup(tmp_path)
        resolver = EntityResolver(graph_store=store)
        assert resolver.resolve("Nonexistent", "thing") is None

    def test_fuzzy_match(self, tmp_path):
        store = self._setup(tmp_path)
        store.add_entity("ChromaDB", "technology")
        resolver = EntityResolver(graph_store=store, fuzzy_threshold=0.7)
        match = resolver.resolve("chromadb", "technology")
        assert match is not None

    def test_merge_entities(self, tmp_path):
        store = self._setup(tmp_path)
        e1 = store.add_entity("Python", "technology", aliases=["Python3"])
        e2 = store.add_entity("python", "technology", aliases=["CPython"])
        resolver = EntityResolver(graph_store=store)
        merged_id = resolver.merge(e1, e2)
        # After merge, the surviving entity should have combined aliases
        entity = store.get_entity(merged_id)
        assert "Python3" in entity["aliases"] or "CPython" in entity["aliases"]
