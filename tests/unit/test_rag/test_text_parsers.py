from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# text parser
# ---------------------------------------------------------------------------

def test_parse_text_basic(tmp_path):
    from homie_core.rag.parsers.text import parse_text
    f = tmp_path / "notes.txt"
    f.write_text("Hello world", encoding="utf-8")
    doc = parse_text(f)
    assert doc.full_text == "Hello world"
    assert doc.source_path == str(f)
    assert doc.metadata["format"] == "text"
    assert doc.metadata["size"] == f.stat().st_size


def test_parse_text_registers_in_registry(tmp_path):
    # importing text.py should side-effect register "text" in PARSER_REGISTRY
    import homie_core.rag.parsers.text  # noqa: F401
    from homie_core.rag.parsers import PARSER_REGISTRY
    assert "text" in PARSER_REGISTRY
    assert "markdown" in PARSER_REGISTRY
    assert "csv" in PARSER_REGISTRY


def test_parse_text_single_block(tmp_path):
    from homie_core.rag.parsers.text import parse_text
    f = tmp_path / "file.txt"
    f.write_text("Line one\nLine two\nLine three", encoding="utf-8")
    doc = parse_text(f)
    assert len(doc.text_blocks) == 1
    assert doc.text_blocks[0].block_type == "paragraph"


def test_parse_text_handles_encoding_errors(tmp_path):
    from homie_core.rag.parsers.text import parse_text
    f = tmp_path / "latin.txt"
    f.write_bytes(b"Caf\xe9")  # latin-1 byte, not valid UTF-8
    doc = parse_text(f)
    assert "Caf" in doc.full_text  # replacement char, but no crash


# ---------------------------------------------------------------------------
# markdown parser
# ---------------------------------------------------------------------------

def test_parse_markdown_headings(tmp_path):
    from homie_core.rag.parsers.text import parse_markdown
    content = "# Title\n\nSome paragraph.\n\n## Section\n\nMore text."
    f = tmp_path / "doc.md"
    f.write_text(content, encoding="utf-8")
    doc = parse_markdown(f)
    headings = [b for b in doc.text_blocks if b.block_type == "heading"]
    assert len(headings) == 2
    assert headings[0].content == "Title"
    assert headings[0].level == 1
    assert headings[1].content == "Section"
    assert headings[1].level == 2


def test_parse_markdown_paragraph_parent_heading(tmp_path):
    from homie_core.rag.parsers.text import parse_markdown
    content = "# Intro\n\nHello there."
    f = tmp_path / "doc.md"
    f.write_text(content, encoding="utf-8")
    doc = parse_markdown(f)
    paragraphs = [b for b in doc.text_blocks if b.block_type == "paragraph"]
    assert len(paragraphs) == 1
    assert paragraphs[0].parent_heading == "Intro"


def test_parse_markdown_metadata(tmp_path):
    from homie_core.rag.parsers.text import parse_markdown
    content = "# Title\n\nBody."
    f = tmp_path / "doc.md"
    f.write_text(content, encoding="utf-8")
    doc = parse_markdown(f)
    assert doc.metadata["format"] == "markdown"
    assert "size" in doc.metadata
    assert doc.source_path == str(f)


def test_parse_markdown_empty_file(tmp_path):
    from homie_core.rag.parsers.text import parse_markdown
    f = tmp_path / "empty.md"
    f.write_text("", encoding="utf-8")
    doc = parse_markdown(f)
    # empty content — should have one fallback block
    assert len(doc.text_blocks) == 1


def test_parse_markdown_no_headings(tmp_path):
    from homie_core.rag.parsers.text import parse_markdown
    f = tmp_path / "plain.md"
    f.write_text("Just a plain paragraph.", encoding="utf-8")
    doc = parse_markdown(f)
    assert len(doc.text_blocks) == 1
    assert doc.text_blocks[0].block_type == "paragraph"
    assert doc.text_blocks[0].parent_heading is None


# ---------------------------------------------------------------------------
# csv parser
# ---------------------------------------------------------------------------

def test_parse_csv_basic(tmp_path):
    from homie_core.rag.parsers.text import parse_csv
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
    doc = parse_csv(f)
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.headers == ["name", "age"]
    assert table.rows == [["Alice", "30"], ["Bob", "25"]]
    assert doc.metadata["format"] == "csv"
    assert doc.metadata["rows"] == 2
    assert doc.metadata["columns"] == 2


def test_parse_csv_full_text_contains_data(tmp_path):
    from homie_core.rag.parsers.text import parse_csv
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2", encoding="utf-8")
    doc = parse_csv(f)
    assert "a" in doc.full_text
    assert "1" in doc.full_text


def test_parse_csv_empty_file(tmp_path):
    from homie_core.rag.parsers.text import parse_csv
    f = tmp_path / "empty.csv"
    f.write_text("", encoding="utf-8")
    doc = parse_csv(f)
    assert doc.source_path == str(f)
    # no crash — empty document is valid
    assert doc.tables == [] or doc.tables[0].headers == []


def test_parse_csv_source_path(tmp_path):
    from homie_core.rag.parsers.text import parse_csv
    f = tmp_path / "data.csv"
    f.write_text("x\n1", encoding="utf-8")
    doc = parse_csv(f)
    assert doc.source_path == str(f)


def test_parse_csv_headers_only(tmp_path):
    from homie_core.rag.parsers.text import parse_csv
    f = tmp_path / "headers.csv"
    f.write_text("col1,col2,col3", encoding="utf-8")
    doc = parse_csv(f)
    assert doc.tables[0].headers == ["col1", "col2", "col3"]
    assert doc.tables[0].rows == []
    assert doc.metadata["rows"] == 0
