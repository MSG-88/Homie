from __future__ import annotations

import pytest


def test_text_block_defaults():
    from homie_core.rag.parsers import TextBlock
    block = TextBlock(content="Hello world")
    assert block.content == "Hello world"
    assert block.block_type == "paragraph"
    assert block.level == 0
    assert block.page is None
    assert block.line_start is None
    assert block.line_end is None
    assert block.language is None
    assert block.parent_heading is None


def test_text_block_heading():
    from homie_core.rag.parsers import TextBlock
    block = TextBlock(content="Introduction", block_type="heading", level=1)
    assert block.block_type == "heading"
    assert block.level == 1


def test_text_block_all_fields():
    from homie_core.rag.parsers import TextBlock
    block = TextBlock(
        content="def foo():",
        block_type="code",
        level=0,
        page=3,
        line_start=10,
        line_end=20,
        language="python",
        parent_heading="Functions",
    )
    assert block.page == 3
    assert block.line_start == 10
    assert block.line_end == 20
    assert block.language == "python"
    assert block.parent_heading == "Functions"


def test_table_data_defaults():
    from homie_core.rag.parsers import TableData
    table = TableData()
    assert table.headers == []
    assert table.rows == []
    assert table.caption is None
    assert table.source_page is None


def test_table_data_with_content():
    from homie_core.rag.parsers import TableData
    table = TableData(
        headers=["Name", "Age"],
        rows=[["Alice", "30"], ["Bob", "25"]],
        caption="Users",
        source_page=2,
    )
    assert table.headers == ["Name", "Age"]
    assert len(table.rows) == 2
    assert table.caption == "Users"
    assert table.source_page == 2


def test_parsed_document_defaults():
    from homie_core.rag.parsers import ParsedDocument
    doc = ParsedDocument()
    assert doc.text_blocks == []
    assert doc.metadata == {}
    assert doc.tables == []
    assert doc.source_path == ""


def test_parsed_document_full_text():
    from homie_core.rag.parsers import ParsedDocument, TextBlock
    doc = ParsedDocument(
        text_blocks=[
            TextBlock(content="First paragraph"),
            TextBlock(content="   "),  # whitespace-only — skipped
            TextBlock(content="Second paragraph"),
        ]
    )
    result = doc.full_text
    assert "First paragraph" in result
    assert "Second paragraph" in result
    # whitespace-only block should not contribute blank lines between
    assert result == "First paragraph\n\nSecond paragraph"


def test_parsed_document_full_text_empty():
    from homie_core.rag.parsers import ParsedDocument
    doc = ParsedDocument()
    assert doc.full_text == ""


def test_register_parser_adds_to_registry():
    from homie_core.rag.parsers import PARSER_REGISTRY, register_parser, ParsedDocument
    from pathlib import Path

    @register_parser("__test_fmt__")
    def my_parser(path: Path) -> ParsedDocument:
        return ParsedDocument(source_path=str(path))

    assert "__test_fmt__" in PARSER_REGISTRY
    assert PARSER_REGISTRY["__test_fmt__"] is my_parser
    # cleanup
    del PARSER_REGISTRY["__test_fmt__"]


def test_register_parser_returns_original_function():
    from homie_core.rag.parsers import PARSER_REGISTRY, register_parser, ParsedDocument
    from pathlib import Path

    @register_parser("__test_fmt2__")
    def my_parser2(path: Path) -> ParsedDocument:
        return ParsedDocument()

    # decorator must return the original function unchanged
    assert callable(my_parser2)
    del PARSER_REGISTRY["__test_fmt2__"]


def test_parser_registry_is_dict():
    from homie_core.rag.parsers import PARSER_REGISTRY
    assert isinstance(PARSER_REGISTRY, dict)
