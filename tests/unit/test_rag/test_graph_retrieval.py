"""Tests for graph-expanded retrieval — TDD for SP3 Advanced Retrieval."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from homie_core.knowledge.graph import KnowledgeGraph
from homie_core.knowledge.models import Entity, Relationship
from homie_core.rag.graph_retrieval import RetrievalChunk, graph_expand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(content: str, source: str = "test", score: float = 1.0) -> RetrievalChunk:
    return RetrievalChunk(content=content, source=source, score=score)


def _make_graph(tmp_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# RetrievalChunk dataclass
# ---------------------------------------------------------------------------

class TestRetrievalChunk:
    def test_default_score_is_zero(self):
        chunk = RetrievalChunk(content="hello", source="test")
        assert chunk.score == 0.0

    def test_default_metadata_is_empty_dict(self):
        chunk = RetrievalChunk(content="hello", source="test")
        assert chunk.metadata == {}

    def test_metadata_not_shared_between_instances(self):
        c1 = RetrievalChunk(content="a", source="s")
        c2 = RetrievalChunk(content="b", source="s")
        c1.metadata["key"] = "val"
        assert "key" not in c2.metadata


# ---------------------------------------------------------------------------
# graph_expand — no graph
# ---------------------------------------------------------------------------

class TestGraphExpandNoGraph:
    def test_no_graph_returns_chunks_unchanged(self):
        chunks = [_make_chunk("text")]
        result = graph_expand("query", chunks, graph=None)
        assert result == chunks

    def test_no_graph_empty_chunks_returns_empty(self):
        result = graph_expand("query", [], graph=None)
        assert result == []

    def test_no_chunks_with_graph_returns_empty(self, tmp_path):
        graph = _make_graph(tmp_path)
        result = graph_expand("some query", [], graph=graph)
        assert result == []


# ---------------------------------------------------------------------------
# graph_expand — with graph
# ---------------------------------------------------------------------------

class TestGraphExpandWithGraph:
    def test_no_entities_mentioned_returns_unchanged(self, tmp_path):
        graph = _make_graph(tmp_path)
        # Add an entity that is NOT in the query
        entity = Entity(name="Alice", entity_type="person")
        graph.add_entity(entity)

        chunks = [_make_chunk("something unrelated")]
        result = graph_expand("totally unrelated query", chunks, graph=graph)
        # No entity names appear in query — unchanged
        assert len(result) == len(chunks)

    def test_entity_found_adds_context_chunk(self, tmp_path):
        graph = _make_graph(tmp_path)
        entity = Entity(name="Homie", entity_type="project", confidence=0.9)
        graph.add_entity(entity)

        chunks = [_make_chunk("base chunk")]
        result = graph_expand("Tell me about Homie", chunks, graph=graph)
        assert len(result) == 2
        # The extra chunk is from the knowledge graph
        sources = [c.source for c in result]
        assert any("knowledge_graph" in s for s in sources)

    def test_entity_context_chunk_score_is_confidence_times_0_8(self, tmp_path):
        graph = _make_graph(tmp_path)
        confidence = 0.75
        entity = Entity(name="Python", entity_type="tool", confidence=confidence)
        graph.add_entity(entity)

        chunks = [_make_chunk("base")]
        result = graph_expand("Python programming", chunks, graph=graph)
        extra = [c for c in result if "knowledge_graph" in c.source]
        assert len(extra) == 1
        assert abs(extra[0].score - confidence * 0.8) < 1e-9

    def test_respects_budget_limit(self, tmp_path):
        graph = _make_graph(tmp_path)
        # Add 5 entities that will be mentioned in the query
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        for name in names:
            graph.add_entity(Entity(name=name, entity_type="concept"))

        query = "Alpha Beta Gamma Delta Epsilon are concepts"
        chunks = [_make_chunk("base")]
        result = graph_expand(query, chunks, graph=graph, budget=2)
        extra = [c for c in result if "knowledge_graph" in c.source]
        assert len(extra) <= 2

    def test_expanded_chunk_metadata_has_entity_id(self, tmp_path):
        graph = _make_graph(tmp_path)
        entity = Entity(name="Django", entity_type="tool")
        graph.add_entity(entity)

        chunks = [_make_chunk("base")]
        result = graph_expand("Django framework", chunks, graph=graph)
        extra = [c for c in result if "knowledge_graph" in c.source]
        assert extra[0].metadata["entity_id"] == entity.id

    def test_expanded_chunk_metadata_has_entity_type(self, tmp_path):
        graph = _make_graph(tmp_path)
        entity = Entity(name="FastAPI", entity_type="tool")
        graph.add_entity(entity)

        chunks = [_make_chunk("base")]
        result = graph_expand("FastAPI endpoint", chunks, graph=graph)
        extra = [c for c in result if "knowledge_graph" in c.source]
        assert extra[0].metadata["entity_type"] == "tool"

    def test_source_format_is_knowledge_graph_colon_name(self, tmp_path):
        graph = _make_graph(tmp_path)
        entity = Entity(name="Redis", entity_type="tool")
        graph.add_entity(entity)

        chunks = [_make_chunk("base")]
        result = graph_expand("Redis cache", chunks, graph=graph)
        extra = [c for c in result if "knowledge_graph" in c.source]
        assert extra[0].source == f"knowledge_graph:{entity.name}"

    def test_original_chunks_preserved_at_start(self, tmp_path):
        graph = _make_graph(tmp_path)
        entity = Entity(name="Celery", entity_type="tool")
        graph.add_entity(entity)

        base = _make_chunk("base chunk content")
        result = graph_expand("Celery task queue", [base], graph=graph)
        # Original chunk comes first
        assert result[0] is base

    def test_default_budget_is_3(self, tmp_path):
        graph = _make_graph(tmp_path)
        # Add 5 entities matching the query
        for name in ["One", "Two", "Three", "Four", "Five"]:
            graph.add_entity(Entity(name=name, entity_type="concept"))

        query = "One Two Three Four Five"
        chunks = [_make_chunk("base")]
        # Call with no explicit budget — should use default of 3
        result = graph_expand(query, chunks, graph=graph)
        extra = [c for c in result if "knowledge_graph" in c.source]
        assert len(extra) <= 3
