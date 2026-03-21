"""Tests for ContextAssembler — TDD for SP3 Advanced Retrieval."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from homie_core.knowledge.graph import KnowledgeGraph
from homie_core.knowledge.models import Entity
from homie_core.rag.context_assembler import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(content: str, source: str = "doc.txt", score: float = 1.0) -> dict:
    return {"content": content, "source": source, "score": score}


def _make_graph(tmp_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph(tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Basic assembly
# ---------------------------------------------------------------------------

class TestContextAssemblerBasic:
    def test_empty_chunks_returns_empty_string(self):
        ca = ContextAssembler()
        result = ca.assemble("query", [], [], None)
        assert result == ""

    def test_single_chunk_included_in_output(self):
        ca = ContextAssembler()
        result = ca.assemble("query", [_chunk("hello world")], [], None)
        assert "hello world" in result

    def test_source_attribution_in_output(self):
        ca = ContextAssembler()
        result = ca.assemble("query", [_chunk("text", source="notes.md")], [], None)
        assert "notes.md" in result

    def test_multiple_chunks_all_included_within_budget(self):
        ca = ContextAssembler()
        chunks = [_chunk(f"content {i}", source=f"file{i}.txt") for i in range(3)]
        result = ca.assemble("query", chunks, [], None, token_budget=4096)
        # All 3 should fit within a large budget
        for i in range(3):
            assert f"content {i}" in result

    def test_returns_string(self):
        ca = ContextAssembler()
        result = ca.assemble("query", [_chunk("data")], [], None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

class TestContextAssemblerBudget:
    def test_token_budget_respected(self):
        ca = ContextAssembler()
        # Each chunk is ~50 chars; budget is 10 tokens = 40 chars
        chunks = [_chunk("x" * 200, source="src") for _ in range(10)]
        result = ca.assemble("query", chunks, [], None, token_budget=10)
        # Result must not exceed budget * 4 chars (plus some overhead for formatting)
        # The key test: not all 10 chunks are included
        assert result.count("[Source:") < 10

    def test_tier1_gets_most_of_budget(self):
        ca = ContextAssembler()
        # Large chunks — with 60% of budget reserved for tier1
        large_chunk = _chunk("A" * 1000, source="big.txt")
        result = ca.assemble("query", [large_chunk], [], None, token_budget=4096)
        assert "A" * 100 in result  # significant portion of chunk present


# ---------------------------------------------------------------------------
# Tier 3: Conversation
# ---------------------------------------------------------------------------

class TestContextAssemblerConversation:
    def test_conversation_included_as_tier3(self):
        ca = ContextAssembler()
        conv = [
            {"role": "user", "content": "what is homie?"},
            {"role": "assistant", "content": "Homie is an AI assistant."},
        ]
        result = ca.assemble("query", [], conv, None)
        assert "what is homie?" in result or "Homie is an AI assistant" in result

    def test_only_last_6_messages_used(self):
        ca = ContextAssembler()
        # 10 messages — only the last 6 should appear
        conv = [{"role": "user", "content": f"message {i}"} for i in range(10)]
        result = ca.assemble("query", [], conv, None, token_budget=4096)
        # Last 6 messages (4-9) should be present; first 4 might not be
        for i in range(4, 10):
            assert f"message {i}" in result

    def test_empty_conversation_no_tier3_section(self):
        ca = ContextAssembler()
        result = ca.assemble("query", [_chunk("data")], [], None)
        assert "Recent Conversation" not in result

    def test_conversation_section_label(self):
        ca = ContextAssembler()
        conv = [{"role": "user", "content": "hello"}]
        result = ca.assemble("query", [], conv, None)
        assert "Recent Conversation" in result


# ---------------------------------------------------------------------------
# Tier 4: User profile
# ---------------------------------------------------------------------------

class TestContextAssemblerUserProfile:
    def test_user_profile_included(self):
        ca = ContextAssembler()
        profile = {"name": "Alice", "occupation": "Engineer"}
        result = ca.assemble("query", [], [], profile, token_budget=4096)
        assert "Alice" in result
        assert "Engineer" in result

    def test_user_profile_section_label(self):
        ca = ContextAssembler()
        profile = {"city": "London"}
        result = ca.assemble("query", [], [], profile, token_budget=4096)
        assert "User Profile" in result

    def test_none_profile_no_user_profile_section(self):
        ca = ContextAssembler()
        result = ca.assemble("query", [_chunk("data")], [], None)
        assert "User Profile" not in result

    def test_empty_profile_values_excluded(self):
        ca = ContextAssembler()
        profile = {"name": "Bob", "empty_field": ""}
        result = ca.assemble("query", [], [], profile, token_budget=4096)
        assert "Bob" in result
        assert "empty_field" not in result


# ---------------------------------------------------------------------------
# Tier 2: Graph entity context
# ---------------------------------------------------------------------------

class TestContextAssemblerGraphTier:
    def test_graph_entity_context_included(self, tmp_path):
        graph = _make_graph(tmp_path)
        entity = Entity(name="Homie", entity_type="project")
        graph.add_entity(entity)
        ca = ContextAssembler(graph=graph)
        result = ca.assemble("Tell me about Homie", [], [], None, token_budget=4096)
        assert "Homie" in result

    def test_entity_label_in_output(self, tmp_path):
        graph = _make_graph(tmp_path)
        entity = Entity(name="Redis", entity_type="tool")
        graph.add_entity(entity)
        ca = ContextAssembler(graph=graph)
        result = ca.assemble("Redis cache query", [], [], None, token_budget=4096)
        assert "Entity:" in result or "Redis" in result

    def test_no_graph_no_entity_section(self):
        ca = ContextAssembler(graph=None)
        result = ca.assemble("query", [_chunk("data")], [], None)
        assert "Entity:" not in result


# ---------------------------------------------------------------------------
# Lost-in-middle mitigation
# ---------------------------------------------------------------------------

class TestLostInMiddleMitigation:
    def test_multiple_tier1_chunks_split_between_start_and_end(self):
        ca = ContextAssembler()
        # Create 4 tier1 chunks — should be split: 2 at start, 2 at end
        chunks = [_chunk(f"tier1_chunk_{i}", source=f"doc{i}.txt") for i in range(4)]
        conv = [{"role": "user", "content": "some conversation context here"}]
        result = ca.assemble("query", chunks, conv, None, token_budget=4096)
        # All chunks are present
        for i in range(4):
            assert f"tier1_chunk_{i}" in result
        # Conversation is sandwiched between the tier1 chunks
        # Find positions
        pos_conv = result.find("Recent Conversation")
        pos_chunk_0 = result.find("tier1_chunk_0")
        pos_chunk_2 = result.find("tier1_chunk_2")
        assert pos_chunk_0 < pos_conv < pos_chunk_2

    def test_single_tier1_chunk_at_start(self):
        ca = ContextAssembler()
        chunks = [_chunk("only chunk")]
        conv = [{"role": "user", "content": "conversation"}]
        result = ca.assemble("query", chunks, conv, None, token_budget=4096)
        pos_chunk = result.find("only chunk")
        pos_conv = result.find("Recent Conversation")
        assert pos_chunk < pos_conv

    def test_no_chunks_with_conv_and_profile(self):
        ca = ContextAssembler()
        conv = [{"role": "user", "content": "hello"}]
        profile = {"name": "Bob"}
        result = ca.assemble("query", [], conv, profile, token_budget=4096)
        assert "hello" in result or "Bob" in result
