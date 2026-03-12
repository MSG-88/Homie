"""AI tool wrappers for folder awareness integration.

Registers 3 tools with the ToolRegistry so the Brain can interact
with watched folders via the agentic loop.
"""
from __future__ import annotations

import json

from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "..."
    return text


def register_folder_tools(registry: ToolRegistry, folder_service) -> None:
    """Register all folder tools with the tool registry."""

    def tool_folder_search(query: str, folder: str = "", max_results: str = "10") -> str:
        try:
            limit = int(max_results)
        except (ValueError, TypeError):
            limit = 10
        results = folder_service.search(query, folder=folder or None, max_results=limit)
        return _truncate(json.dumps([
            {
                "source": f.source,
                "content_type": f.content_type,
                "summary": f.summary,
                "topics": f.topics,
                "size": f.size,
                "modified_at": f.modified_at,
            }
            for f in results
        ]))

    registry.register(Tool(
        name="folder_search",
        description="Search indexed files in watched folders by filename or content.",
        params=[
            ToolParam(name="query", description="Search query (matches file paths and content summaries)", type="string"),
            ToolParam(name="folder", description="Limit search to a specific folder path", type="string", required=False, default=""),
            ToolParam(name="max_results", description="Maximum results to return", type="string", required=False, default="10"),
        ],
        execute=tool_folder_search,
        category="folders",
    ))

    def tool_folder_summary(folder: str = "") -> str:
        result = folder_service.get_summary(folder=folder or None)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="folder_summary",
        description="Get overview of watched folders: file counts, types, recent changes.",
        params=[
            ToolParam(name="folder", description="Specific folder path (or empty for all)", type="string", required=False, default=""),
        ],
        execute=tool_folder_summary,
        category="folders",
    ))

    def tool_folder_list_watches() -> str:
        watches = folder_service.list_watches()
        return _truncate(json.dumps(watches))

    registry.register(Tool(
        name="folder_list_watches",
        description="List all watched folders with their status and file counts.",
        params=[],
        execute=tool_folder_list_watches,
        category="folders",
    ))
