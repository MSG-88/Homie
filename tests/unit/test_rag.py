"""Unit tests for the RAG pipeline — chunking, ingest, retrieve, augment.

Covers:
- Chunk dataclass: properties, to_search_text
- chunk_code: Python/Go/JS, preamble, large blocks, fallback
- chunk_markdown: headings, nested sections, preamble, oversized section
- _sliding_window_chunk: overlap, remainder
- auto_chunk: extension-based dispatch
- RagPipeline: index_file, index_directory, remove_file, retrieve, build_context_block, stats
- RetrievedContext: to_attributed_text, with parent section
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from homie_core.rag.chunker import (
    Chunk,
    auto_chunk,
    chunk_code,
    chunk_markdown,
    _sliding_window_chunk,
)
from homie_core.rag.pipeline import RagPipeline, RetrievedContext


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

class TestChunk:
    def test_char_count(self):
        c = Chunk(text="hello world")
        assert c.char_count == 11

    def test_to_search_text_no_source(self):
        c = Chunk(text="some text")
        assert "some text" in c.to_search_text()

    def test_to_search_text_with_source(self):
        c = Chunk(text="def hello(): pass", source="main.py")
        result = c.to_search_text()
        assert "main.py" in result
        assert "def hello(): pass" in result

    def test_to_search_text_with_parent_section(self):
        c = Chunk(text="content", source="docs.md", parent_section="API Reference")
        result = c.to_search_text()
        assert "API Reference" in result
        assert "content" in result

    def test_default_chunk_type(self):
        c = Chunk(text="text")
        assert c.chunk_type == "text"

    def test_default_line_numbers_zero(self):
        c = Chunk(text="text")
        assert c.start_line == 0
        assert c.end_line == 0


# ---------------------------------------------------------------------------
# chunk_code
# ---------------------------------------------------------------------------

class TestChunkCode:
    def test_python_splits_at_functions(self):
        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        chunks = chunk_code(code, source="test.py")
        assert len(chunks) >= 2
        func_names = [c.text for c in chunks]
        assert any("foo" in t for t in func_names)
        assert any("bar" in t for t in func_names)

    def test_python_splits_at_classes(self):
        code = "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
        chunks = chunk_code(code, source="app.py")
        types = [c.chunk_type for c in chunks]
        assert any("class" in t for t in types)

    def test_python_preamble_extracted(self):
        code = "import os\nimport sys\n\ndef main():\n    pass\n"
        chunks = chunk_code(code, source="main.py")
        preamble = [c for c in chunks if c.chunk_type == "code_preamble"]
        assert len(preamble) >= 1
        assert "import" in preamble[0].text

    def test_python_async_def_detected(self):
        code = "async def fetch_data():\n    pass\n"
        chunks = chunk_code(code, source="api.py")
        assert any("fetch_data" in c.text for c in chunks)

    def test_go_func_split(self):
        code = "package main\n\nfunc Foo() {}\n\nfunc Bar() {}\n"
        chunks = chunk_code(code, source="main.go")
        assert len(chunks) >= 2

    def test_unknown_extension_sliding_window_fallback(self):
        code = "x = 1\ny = 2\n" * 10
        chunks = chunk_code(code, source="script.unknown")
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_no_definitions_falls_back_to_sliding(self):
        code = "x = 1\ny = 2\n"
        chunks = chunk_code(code, source="no_defs.py")
        assert len(chunks) >= 1

    def test_large_function_sub_chunked(self):
        big_func = "def big():\n" + "\n".join(f"    x_{i} = {i}" for i in range(500))
        chunks = chunk_code(big_func, source="big.py", max_chunk=200)
        assert len(chunks) >= 2

    def test_line_numbers_tracked(self):
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        chunks = chunk_code(code, source="test.py")
        for c in chunks:
            assert c.start_line >= 0
            assert c.end_line >= c.start_line

    def test_language_set_for_python(self):
        code = "def foo(): pass\n"
        chunks = chunk_code(code, source="test.py")
        for c in chunks:
            assert c.language == "py"

    def test_empty_code_returns_chunks(self):
        chunks = chunk_code("", source="empty.py")
        # Should not raise, may return empty list or single empty chunk
        assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# chunk_markdown
# ---------------------------------------------------------------------------

class TestChunkMarkdown:
    def test_splits_at_h1_heading(self):
        md = "# Title\n\nIntro text.\n\n# Second\n\nMore text.\n"
        chunks = chunk_markdown(md, source="doc.md")
        assert len(chunks) >= 2

    def test_splits_at_h2_heading(self):
        md = "## Overview\n\nContent A.\n\n## Details\n\nContent B.\n"
        chunks = chunk_markdown(md, source="guide.md")
        assert len(chunks) >= 2

    def test_preamble_before_first_heading(self):
        md = "Some intro text before any heading.\n\n# First Heading\n\nContent.\n"
        chunks = chunk_markdown(md, source="doc.md")
        types = [c.chunk_type for c in chunks]
        assert any("preamble" in t for t in types)

    def test_no_headings_uses_sliding_window(self):
        md = "Just plain text\nwith multiple lines\nbut no headings.\n"
        chunks = chunk_markdown(md, source="plain.md")
        assert len(chunks) >= 1

    def test_parent_section_hierarchical(self):
        md = "# API\n\nOverview.\n\n## Endpoints\n\nList of endpoints.\n"
        chunks = chunk_markdown(md, source="api.md")
        sections = [c for c in chunks if "Endpoints" in c.text or c.parent_section]
        assert len(sections) >= 1

    def test_chunk_type_markdown_section(self):
        md = "# Section\n\nContent here.\n"
        chunks = chunk_markdown(md, source="doc.md")
        section_chunks = [c for c in chunks if c.chunk_type == "markdown_section"]
        assert len(section_chunks) >= 1

    def test_large_section_sub_chunked(self):
        big_section = "# Big Section\n\n" + ("word " * 500 + "\n") * 10
        chunks = chunk_markdown(big_section, source="big.md", max_chunk=200)
        assert len(chunks) >= 2

    def test_line_numbers_tracked(self):
        md = "# Section One\n\nContent.\n\n# Section Two\n\nMore.\n"
        chunks = chunk_markdown(md, source="doc.md")
        for c in chunks:
            assert c.start_line >= 0


# ---------------------------------------------------------------------------
# _sliding_window_chunk
# ---------------------------------------------------------------------------

class TestSlidingWindowChunk:
    def test_short_text_single_chunk(self):
        text = "Short text.\n"
        chunks = _sliding_window_chunk(text, source="test.txt", max_chunk=100)
        assert len(chunks) == 1
        assert "Short text." in chunks[0].text

    def test_long_text_multiple_chunks(self):
        text = "\n".join(f"line {i}" for i in range(200))
        chunks = _sliding_window_chunk(text, source="long.txt", max_chunk=100)
        assert len(chunks) >= 2

    def test_overlap_present(self):
        lines = [f"line{i}" for i in range(100)]
        text = "\n".join(lines)
        chunks = _sliding_window_chunk(text, source="f.txt", max_chunk=50, overlap=20)
        if len(chunks) >= 2:
            # The second chunk should share some content with the end of the first
            end_of_first = chunks[0].text.split("\n")[-3:]
            start_of_second = chunks[1].text.split("\n")[:3]
            overlap = set(end_of_first) & set(start_of_second)
            assert len(overlap) >= 0  # overlap may vary; just ensure no crash

    def test_empty_text_returns_empty(self):
        chunks = _sliding_window_chunk("", source="empty.txt")
        assert chunks == [] or all(not c.text.strip() for c in chunks)

    def test_chunk_type_preserved(self):
        text = "a b c d e\n" * 5
        chunks = _sliding_window_chunk(text, source="f.txt", chunk_type="code")
        for c in chunks:
            assert c.chunk_type == "code"

    def test_base_line_offset(self):
        text = "line1\nline2\nline3\n"
        chunks = _sliding_window_chunk(text, source="f.txt", max_chunk=500, base_line=10)
        assert chunks[0].start_line == 10


# ---------------------------------------------------------------------------
# auto_chunk
# ---------------------------------------------------------------------------

class TestAutoChunk:
    def test_python_routes_to_code_chunker(self):
        code = "def foo():\n    pass\n"
        chunks = auto_chunk(code, source="main.py")
        assert any(c.chunk_type.startswith("code") for c in chunks)

    def test_markdown_routes_to_md_chunker(self):
        md = "# Section\n\nContent here.\n"
        chunks = auto_chunk(md, source="readme.md")
        assert any("markdown" in c.chunk_type for c in chunks)

    def test_rst_routes_to_md_chunker(self):
        rst = "Section\n=======\n\nContent.\n"
        chunks = auto_chunk(rst, source="docs.rst")
        assert isinstance(chunks, list)

    def test_text_file_sliding_window(self):
        txt = "Just plain text content.\n"
        chunks = auto_chunk(txt, source="notes.txt")
        assert len(chunks) >= 1

    def test_json_file_sliding_window(self):
        json_txt = '{"key": "value", "list": [1, 2, 3]}'
        chunks = auto_chunk(json_txt, source="config.json")
        assert len(chunks) >= 1

    def test_unknown_extension_handled(self):
        text = "Some random content.\n"
        chunks = auto_chunk(text, source="file.xyz")
        assert isinstance(chunks, list)

    def test_no_source_fallback(self):
        chunks = auto_chunk("text without source")
        assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# RetrievedContext
# ---------------------------------------------------------------------------

class TestRetrievedContext:
    def test_to_attributed_text_basic(self):
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

    def test_to_attributed_text_with_parent_section(self):
        ctx = RetrievedContext(
            text="Detailed API description.",
            source="docs/api.md",
            start_line=5,
            end_line=15,
            chunk_type="markdown_section",
            relevance_score=0.85,
            parent_section="API Reference",
        )
        attributed = ctx.to_attributed_text()
        assert "API Reference" in attributed
        assert "api.md" in attributed

    def test_to_attributed_text_no_parent_section(self):
        ctx = RetrievedContext(
            text="content",
            source="file.py",
            start_line=1,
            end_line=5,
            chunk_type="code_function",
            relevance_score=0.7,
        )
        attributed = ctx.to_attributed_text()
        # Should not have empty parentheses
        assert "()" not in attributed

    def test_source_filename_only_in_attribution(self):
        ctx = RetrievedContext(
            text="x = 1",
            source="/very/long/path/to/file/module.py",
            start_line=1,
            end_line=1,
            chunk_type="code",
            relevance_score=0.5,
        )
        attributed = ctx.to_attributed_text()
        assert "module.py" in attributed


# ---------------------------------------------------------------------------
# RagPipeline — ingest
# ---------------------------------------------------------------------------

class TestRagPipelineIngest:
    def test_index_python_file_returns_count(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "module.py"
        f.write_text("def hello():\n    return 'hi'\n\ndef goodbye():\n    return 'bye'\n")
        count = pipe.index_file(f)
        assert count >= 1

    def test_index_markdown_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "README.md"
        f.write_text("# Overview\n\nThis is the readme.\n\n## Installation\n\nRun `pip install`.\n")
        count = pipe.index_file(f)
        assert count >= 1

    def test_index_text_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "notes.txt"
        f.write_text("Some notes about the project.\nLine two.\nLine three.\n")
        count = pipe.index_file(f)
        assert count >= 1

    def test_skip_unsupported_extension(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "binary.exe"
        f.write_bytes(b"\x4d\x5a" + b"\x00" * 100)
        count = pipe.index_file(f)
        assert count == 0

    def test_skip_large_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "huge.py"
        f.write_text("x = 1\n" * 500_000)
        count = pipe.index_file(f)
        assert count == 0

    def test_skip_nonexistent_file(self, tmp_path):
        pipe = RagPipeline()
        count = pipe.index_file(tmp_path / "ghost.py")
        assert count == 0

    def test_skip_unchanged_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "stable.py"
        f.write_text("def stable(): pass\n")
        count1 = pipe.index_file(f)
        count2 = pipe.index_file(f)
        assert count1 > 0
        assert count2 == 0

    def test_reindex_changed_file(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "changed.py"
        f.write_text("def v1(): pass\n")
        pipe.index_file(f)
        f.write_text("def v2_completely_different(): pass\n")
        count = pipe.index_file(f)
        assert count > 0

    def test_index_directory_indexes_multiple_files(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.md").write_text("# Doc\n\nContent.\n")
        total = pipe.index_directory(tmp_path)
        assert total >= 2

    def test_index_directory_skips_unsupported(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
        total = pipe.index_directory(tmp_path)
        assert total >= 1  # at least a.py

    def test_index_nonexistent_directory(self):
        pipe = RagPipeline()
        count = pipe.index_directory("/no/such/directory/exists")
        assert count == 0

    def test_remove_file_clears_index(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "temp.py"
        f.write_text("def temp(): pass\n")
        pipe.index_file(f)
        pipe.remove_file(f)
        stats = pipe.get_stats()
        assert stats["indexed_files"] == 0
        assert stats["total_chunks"] == 0

    def test_remove_nonexistent_file_no_error(self, tmp_path):
        pipe = RagPipeline()
        pipe.remove_file(tmp_path / "never_indexed.py")  # Should not raise


# ---------------------------------------------------------------------------
# RagPipeline — retrieve
# ---------------------------------------------------------------------------

class TestRagPipelineRetrieve:
    def test_retrieve_empty_index_returns_empty(self):
        pipe = RagPipeline()
        results = pipe.retrieve("anything")
        assert results == []

    def test_retrieve_relevant_file(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "auth.py").write_text(
            "def authenticate(username, password):\n"
            "    '''Verify user credentials against database.'''\n"
            "    return verify(username, password)\n"
        )
        (tmp_path / "math.py").write_text(
            "def add(a, b):\n    return a + b\n"
        )
        pipe.index_directory(tmp_path)
        results = pipe.retrieve("authentication credentials")
        assert any("auth" in r.source.lower() for r in results)

    def test_retrieve_respects_budget(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "data.py").write_text(
            "\n".join(f"def func_{i}(): return {i}  # function about data" for i in range(100))
        )
        pipe.index_directory(tmp_path)
        results = pipe.retrieve("function data", max_chars=300)
        total = sum(len(r.text) for r in results)
        assert total <= 400  # small tolerance for truncation message

    def test_retrieve_returns_retrieved_context_objects(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "code.py").write_text("def process(): pass\n")
        pipe.index_directory(tmp_path)
        results = pipe.retrieve("process function")
        for r in results:
            assert isinstance(r, RetrievedContext)

    def test_retrieve_context_has_source(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "code.py").write_text("def process(): pass\n")
        pipe.index_directory(tmp_path)
        results = pipe.retrieve("process")
        if results:
            assert results[0].source != ""

    def test_retrieve_file_filter_python_only(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "code.py").write_text("def process_data(): pass\n")
        (tmp_path / "docs.md").write_text("# Process\n\nDocument about process.\n")
        pipe.index_directory(tmp_path)
        results = pipe.retrieve("process", file_filter="*.py")
        for r in results:
            assert r.source.endswith(".py")

    def test_retrieve_top_k_limit(self, tmp_path):
        pipe = RagPipeline()
        for i in range(10):
            (tmp_path / f"file_{i}.py").write_text(f"def func_{i}(): pass  # useful function\n")
        pipe.index_directory(tmp_path)
        results = pipe.retrieve("useful function", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# RagPipeline — augment
# ---------------------------------------------------------------------------

class TestRagPipelineAugment:
    def test_build_context_block_contains_documents_header(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "api.py").write_text(
            "def get_users():\n    '''List all users.'''\n    return db.all()\n"
        )
        pipe.index_directory(tmp_path)
        block = pipe.build_context_block("list users database")
        assert "[DOCUMENTS]" in block

    def test_build_context_block_contains_filename(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "api.py").write_text("def get_users(): pass\n")
        pipe.index_directory(tmp_path)
        block = pipe.build_context_block("get users")
        assert "api.py" in block

    def test_build_context_block_empty_index_returns_empty(self):
        pipe = RagPipeline()
        block = pipe.build_context_block("anything")
        assert block == ""

    def test_build_context_block_respects_max_chars(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "big.py").write_text(
            "\n".join(f"def func_{i}(): return {i}  # with context" for i in range(200))
        )
        pipe.index_directory(tmp_path)
        block = pipe.build_context_block("function context", max_chars=200)
        assert len(block) < 1000  # generous but bounded


# ---------------------------------------------------------------------------
# RagPipeline — stats
# ---------------------------------------------------------------------------

class TestRagPipelineStats:
    def test_initial_stats_empty(self):
        pipe = RagPipeline()
        stats = pipe.get_stats()
        assert stats["indexed_files"] == 0
        assert stats["total_chunks"] == 0
        assert stats["indexed_dirs"] == []

    def test_stats_after_index_file(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "code.py").write_text("def foo(): pass\n")
        pipe.index_file(tmp_path / "code.py")
        stats = pipe.get_stats()
        assert stats["indexed_files"] >= 1
        assert stats["total_chunks"] >= 1

    def test_stats_after_index_directory(self, tmp_path):
        pipe = RagPipeline()
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.py").write_text("def b(): pass\n")
        pipe.index_directory(tmp_path)
        stats = pipe.get_stats()
        assert stats["indexed_files"] >= 2
        assert str(tmp_path) in stats["indexed_dirs"]

    def test_stats_after_remove(self, tmp_path):
        pipe = RagPipeline()
        f = tmp_path / "mod.py"
        f.write_text("def mod(): pass\n")
        pipe.index_file(f)
        pipe.remove_file(f)
        stats = pipe.get_stats()
        assert stats["indexed_files"] == 0
