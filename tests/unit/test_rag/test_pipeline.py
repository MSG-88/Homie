"""Tests for the RAG pipeline — ingest, retrieve, augment."""
import pytest
from pathlib import Path
from homie_core.rag.pipeline import RagPipeline, RetrievedContext
from homie_core.knowledge.graph import KnowledgeGraph


# -----------------------------------------------------------------------
# Ingest
# -----------------------------------------------------------------------

class TestIngest:
    def test_index_python_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "main.py"
        f.write_text("def hello():\n    return 'world'\n\ndef goodbye():\n    return 'farewell'\n")

        count = pipe.index_file(f)
        assert count >= 1

    def test_skip_binary_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")
        count = pipe.index_file(f)
        assert count == 0

    def test_skip_large_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "big.py"
        f.write_text("x" * 3_000_000)
        count = pipe.index_file(f)
        assert count == 0

    def test_skip_unchanged_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "stable.py"
        f.write_text("def stable(): pass\n")
        count1 = pipe.index_file(f)
        count2 = pipe.index_file(f)
        assert count1 > 0
        assert count2 == 0  # unchanged

    def test_reindex_changed_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "changing.py"
        f.write_text("def v1(): pass\n")
        pipe.index_file(f)
        f.write_text("def v2(): pass\n")
        count = pipe.index_file(f)
        assert count > 0

    def test_index_directory(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.md").write_text("# Hello\n\nWorld.\n")
        (tmp_path / "c.bin").write_bytes(b"\x00\x01")

        total = pipe.index_directory(tmp_path)
        assert total >= 2  # a.py + b.md

    def test_index_nonexistent_directory(self):
        pipe = RagPipeline()
        count = pipe.index_directory("/nonexistent/path")
        assert count == 0

    def test_remove_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "temp.py"
        f.write_text("def temp(): pass\n")
        pipe.index_file(f)
        pipe.remove_file(f)
        stats = pipe.get_stats()
        assert stats["indexed_files"] == 0


# -----------------------------------------------------------------------
# Retrieve
# -----------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_relevant_chunks(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "auth.py").write_text(
            "def authenticate(username, password):\n"
            "    '''Authenticate a user with credentials.'''\n"
            "    return check_password(username, password)\n"
        )
        (tmp_path / "math.py").write_text(
            "def calculate_tax(income, rate):\n"
            "    return income * rate\n"
        )
        pipe.index_directory(tmp_path)

        results = pipe.retrieve("authentication password")
        assert len(results) >= 1
        assert any("auth" in r.source.lower() for r in results)

    def test_retrieve_with_budget(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "long.py").write_text(
            "\n".join(f"def func_{i}(): pass  # function number {i}" for i in range(50))
        )
        pipe.index_directory(tmp_path)

        results = pipe.retrieve("function", max_chars=200)
        total_chars = sum(len(r.text) for r in results)
        assert total_chars <= 300  # budget + some tolerance for truncation

    def test_retrieve_empty_index(self):
        pipe = RagPipeline()
        results = pipe.retrieve("anything")
        assert results == []

    def test_retrieve_with_file_filter(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "code.py").write_text("def hello(): pass\n")
        (tmp_path / "docs.md").write_text("# Hello\n\nDocumentation.\n")
        pipe.index_directory(tmp_path)

        py_results = pipe.retrieve("hello", file_filter="*.py")
        for r in py_results:
            assert r.source.endswith(".py")


# -----------------------------------------------------------------------
# Augment
# -----------------------------------------------------------------------

class TestAugment:
    def test_build_context_block(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "api.py").write_text(
            "def get_users():\n"
            "    '''List all users from the database.'''\n"
            "    return db.query(User).all()\n"
        )
        pipe.index_directory(tmp_path)

        block = pipe.build_context_block("list users from database")
        assert "[DOCUMENTS]" in block
        assert "api.py" in block

    def test_empty_context_block(self):
        pipe = RagPipeline()
        block = pipe.build_context_block("anything")
        assert block == ""


# -----------------------------------------------------------------------
# RetrievedContext
# -----------------------------------------------------------------------

class TestRetrievedContext:
    def test_to_attributed_text(self):
        ctx = RetrievedContext(
            text="def hello(): pass",
            source="src/main.py",
            start_line=10,
            end_line=12,
            chunk_type="code_function",
            relevance_score=0.9,
        )
        attributed = ctx.to_attributed_text()
        assert "main.py:10-12" in attributed
        assert "def hello(): pass" in attributed

    def test_attributed_with_section(self):
        ctx = RetrievedContext(
            text="Some content",
            source="docs/api.md",
            start_line=5,
            end_line=10,
            chunk_type="markdown_section",
            relevance_score=0.8,
            parent_section="API Reference",
        )
        attributed = ctx.to_attributed_text()
        assert "API Reference" in attributed


# -----------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Knowledge Graph Integration
# -----------------------------------------------------------------------

class TestKnowledgeGraphIntegration:
    def test_pipeline_without_kg_works_normally(self, tmp_path):
        """Existing behaviour is unchanged when knowledge_graph=None."""
        pipe = RagPipeline()
        (tmp_path / "main.py").write_text("import os\ndef hello(): pass\n")
        count = pipe.index_file(tmp_path / "main.py")
        assert count >= 1

    def test_pipeline_with_kg_extracts_imports(self, tmp_path):
        """After indexing a Python file, import-based Tool entities appear in the KG."""
        kg = KnowledgeGraph(tmp_path / "kg.db")
        pipe = RagPipeline(knowledge_graph=kg)
        (tmp_path / "script.py").write_text("import numpy\nimport pandas\n")
        pipe.index_file(tmp_path / "script.py")

        entities = kg.find_entities(entity_type="tool")
        names = [e.name for e in entities]
        assert any("numpy" in n for n in names)
        assert any("pandas" in n for n in names)

    def test_pipeline_with_kg_extracts_emails(self, tmp_path):
        """Email addresses in indexed files become Person entities in the KG."""
        kg = KnowledgeGraph(tmp_path / "kg.db")
        pipe = RagPipeline(knowledge_graph=kg)
        (tmp_path / "readme.txt").write_text(
            "Contact support@example.com for help.\n"
        )
        pipe.index_file(tmp_path / "readme.txt")

        persons = kg.find_entities(entity_type="person")
        assert any("support@example.com" in e.name for e in persons)

    def test_pipeline_kg_error_does_not_break_indexing(self, tmp_path):
        """If the KG raises, index_file should still return chunk count."""
        class BrokenKG:
            def merge_entity(self, e):
                raise RuntimeError("KG failure")
            def add_relationship(self, r):
                raise RuntimeError("KG failure")

        pipe = RagPipeline(knowledge_graph=BrokenKG())
        (tmp_path / "code.py").write_text("import os\n")
        count = pipe.index_file(tmp_path / "code.py")
        assert count >= 1

    def test_pipeline_kg_deduplicates_on_reindex(self, tmp_path):
        """Re-indexing the same file should not duplicate KG entities."""
        kg = KnowledgeGraph(tmp_path / "kg.db")
        pipe = RagPipeline(knowledge_graph=kg)
        f = tmp_path / "module.py"
        f.write_text("import os\n")
        pipe.index_file(f)
        # Force reindex by touching the file with new content
        f.write_text("import os\nimport sys\n")
        pipe.index_file(f)

        os_entities = kg.find_entities(name="os", entity_type="tool")
        assert len(os_entities) == 1  # merge_entity deduplicates


class TestStats:
    def test_initial_stats(self):
        pipe = RagPipeline()
        stats = pipe.get_stats()
        assert stats["indexed_files"] == 0
        assert stats["total_chunks"] == 0
        assert stats["indexed_dirs"] == []

    def test_stats_after_indexing(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "test.py").write_text("def test(): pass\n")
        pipe.index_directory(tmp_path)
        stats = pipe.get_stats()
        assert stats["indexed_files"] >= 1
        assert stats["total_chunks"] >= 1
        assert str(tmp_path) in stats["indexed_dirs"]
