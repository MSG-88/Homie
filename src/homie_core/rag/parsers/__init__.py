from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class TextBlock:
    content: str
    block_type: str = "paragraph"  # heading, paragraph, code, table_cell, caption
    level: int = 0                 # heading level 1-6, or 0
    page: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    language: Optional[str] = None
    parent_heading: Optional[str] = None


@dataclass
class TableData:
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: Optional[str] = None
    source_page: Optional[int] = None


@dataclass
class ParsedDocument:
    text_blocks: list[TextBlock] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    tables: list[TableData] = field(default_factory=list)
    source_path: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.content for b in self.text_blocks if b.content.strip())


# Parser registry
PARSER_REGISTRY: dict[str, Callable[[Path], ParsedDocument]] = {}


def register_parser(format_name: str):
    """Decorator to register a parser function."""
    def decorator(func: Callable[[Path], ParsedDocument]):
        PARSER_REGISTRY[format_name] = func
        return func
    return decorator
