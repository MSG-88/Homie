"""Tests for build_middleware_stack — specifically the knowledge_graph wiring."""
from __future__ import annotations

import pytest

from homie_app.middleware_factory import build_middleware_stack
from homie_core.config import HomieConfig
from homie_core.knowledge.graph import KnowledgeGraph
from homie_core.memory.working import WorkingMemory
from homie_core.middleware.context_enricher import ContextEnricherMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(tmp_path) -> HomieConfig:
    cfg = HomieConfig()
    cfg.storage.path = str(tmp_path)
    return cfg


def _wm() -> WorkingMemory:
    return WorkingMemory()


def _stack_middleware_names(stack) -> list[str]:
    return [type(m).__name__ for m in stack._stack]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildMiddlewareStackDefault:
    def test_returns_middleware_stack(self, tmp_path):
        from homie_core.middleware import MiddlewareStack
        stack = build_middleware_stack(_config(tmp_path), _wm())
        assert isinstance(stack, MiddlewareStack)

    def test_no_knowledge_graph_no_context_enricher(self, tmp_path):
        stack = build_middleware_stack(_config(tmp_path), _wm())
        names = _stack_middleware_names(stack)
        assert "ContextEnricherMiddleware" not in names


class TestBuildMiddlewareStackWithGraph:
    def test_with_graph_context_enricher_present(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "kg.db")
        stack = build_middleware_stack(_config(tmp_path), _wm(), knowledge_graph=graph)
        names = _stack_middleware_names(stack)
        assert "ContextEnricherMiddleware" in names

    def test_context_enricher_has_graph_wired(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "kg.db")
        stack = build_middleware_stack(_config(tmp_path), _wm(), knowledge_graph=graph)
        enrichers = [m for m in stack._stack if isinstance(m, ContextEnricherMiddleware)]
        assert len(enrichers) == 1
        assert enrichers[0]._graph is graph

    def test_none_graph_no_context_enricher(self, tmp_path):
        stack = build_middleware_stack(_config(tmp_path), _wm(), knowledge_graph=None)
        names = _stack_middleware_names(stack)
        assert "ContextEnricherMiddleware" not in names
