from __future__ import annotations
from pathlib import Path
from homie_core.rag.parsers import ParsedDocument, TextBlock, register_parser


@register_parser("text")
def parse_text(path: Path) -> ParsedDocument:
    content = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDocument(
        text_blocks=[TextBlock(content=content, block_type="paragraph")],
        metadata={"format": "text", "size": path.stat().st_size},
        source_path=str(path),
    )


@register_parser("markdown")
def parse_markdown(path: Path) -> ParsedDocument:
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = []
    current_heading = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            current_heading = title
            blocks.append(TextBlock(content=title, block_type="heading", level=level))
        elif stripped:
            blocks.append(TextBlock(content=stripped, block_type="paragraph", parent_heading=current_heading))
    if not blocks:
        blocks.append(TextBlock(content=content, block_type="paragraph"))
    return ParsedDocument(
        text_blocks=blocks,
        metadata={"format": "markdown", "size": path.stat().st_size},
        source_path=str(path),
    )


@register_parser("csv")
def parse_csv(path: Path) -> ParsedDocument:
    import csv
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return ParsedDocument(source_path=str(path))
    headers = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    from homie_core.rag.parsers import TableData
    table = TableData(headers=headers, rows=data_rows)
    text = "\n".join(", ".join(row) for row in rows)
    return ParsedDocument(
        text_blocks=[TextBlock(content=text, block_type="paragraph")],
        tables=[table],
        metadata={"format": "csv", "rows": len(data_rows), "columns": len(headers)},
        source_path=str(path),
    )
