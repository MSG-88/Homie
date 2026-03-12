"""AI tool wrappers for social/messaging integration.

Registers 4 tools with the ToolRegistry so the Brain can interact
with Slack (and future platforms) via the agentic loop.
"""
from __future__ import annotations

import json

from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "..."
    return text


def register_social_tools(registry: ToolRegistry, social_service) -> None:
    """Register all social/messaging tools with the tool registry."""

    def tool_social_channels(platform: str = "all") -> str:
        channels = social_service.list_channels(platform=platform)
        return _truncate(json.dumps(channels))

    registry.register(Tool(
        name="social_channels",
        description="List available channels/conversations on social platforms (Slack, etc.).",
        params=[
            ToolParam(name="platform", description="Platform name or 'all'", type="string", required=False, default="all"),
        ],
        execute=tool_social_channels,
        category="social",
    ))

    def tool_social_messages(channel_id: str, platform: str = "", limit: str = "20") -> str:
        try:
            num_limit = int(limit)
        except (ValueError, TypeError):
            num_limit = 20
        messages = social_service.get_messages(
            channel_id, platform=platform or None, limit=num_limit,
        )
        return _truncate(json.dumps(messages))

    registry.register(Tool(
        name="social_messages",
        description="Get recent messages from a social channel/conversation.",
        params=[
            ToolParam(name="channel_id", description="Channel ID", type="string"),
            ToolParam(name="platform", description="Platform name (optional)", type="string", required=False, default=""),
            ToolParam(name="limit", description="Maximum messages to return", type="string", required=False, default="20"),
        ],
        execute=tool_social_messages,
        category="social",
    ))

    def tool_social_search(query: str, platform: str = "all", limit: str = "10") -> str:
        try:
            num_limit = int(limit)
        except (ValueError, TypeError):
            num_limit = 10
        results = social_service.search(query, platform=platform, limit=num_limit)
        return _truncate(json.dumps(results))

    registry.register(Tool(
        name="social_search",
        description="Search messages across social platforms (Slack, etc.).",
        params=[
            ToolParam(name="query", description="Search query", type="string"),
            ToolParam(name="platform", description="Platform name or 'all'", type="string", required=False, default="all"),
            ToolParam(name="limit", description="Maximum results", type="string", required=False, default="10"),
        ],
        execute=tool_social_search,
        category="social",
    ))

    def tool_social_unread() -> str:
        result = social_service.get_unread()
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="social_unread",
        description="Get unread mentions and direct messages across social platforms.",
        params=[],
        execute=tool_social_unread,
        category="social",
    ))
