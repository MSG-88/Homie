"""AI tool wrappers for financial integration."""
from __future__ import annotations

import json

from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "..."
    return text


def register_financial_tools(registry: ToolRegistry, financial_service) -> None:
    """Register all financial tools with the tool registry."""

    def tool_financial_summary(month: str = "", year: str = "") -> str:
        try:
            y = int(year) if year else None
        except (ValueError, TypeError):
            y = None
        try:
            m = int(month) if month else None
        except (ValueError, TypeError):
            m = None
        result = financial_service.get_summary(year=y, month=m)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="financial_summary",
        description="Get spending overview: monthly totals by category, pending/overdue counts.",
        params=[
            ToolParam(name="month", description="Month number (1-12)", type="string", required=False, default=""),
            ToolParam(name="year", description="Year (e.g. 2026)", type="string", required=False, default=""),
        ],
        execute=tool_financial_summary,
        category="financial",
    ))

    def tool_financial_upcoming(days: str = "7") -> str:
        try:
            d = int(days)
        except (ValueError, TypeError):
            d = 7
        result = financial_service.get_upcoming(days=d)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="financial_upcoming",
        description="List upcoming bills due within N days.",
        params=[
            ToolParam(name="days", description="Number of days to look ahead", type="string", required=False, default="7"),
        ],
        execute=tool_financial_upcoming,
        category="financial",
    ))

    def tool_financial_history(category: str = "", limit: str = "20") -> str:
        try:
            lim = int(limit)
        except (ValueError, TypeError):
            lim = 20
        result = financial_service.get_history(category=category or None, limit=lim)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="financial_history",
        description="Query past financial records, optionally filtered by category.",
        params=[
            ToolParam(name="category", description="Filter by category (bill, subscription, etc.)", type="string", required=False, default=""),
            ToolParam(name="limit", description="Maximum results", type="string", required=False, default="20"),
        ],
        execute=tool_financial_history,
        category="financial",
    ))

    def tool_financial_mark_paid(record_id: str) -> str:
        try:
            rid = int(record_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "Invalid record ID"})
        result = financial_service.mark_paid(rid)
        return json.dumps(result)

    registry.register(Tool(
        name="financial_mark_paid",
        description="Mark a bill as paid by its record ID.",
        params=[
            ToolParam(name="record_id", description="Financial record ID", type="string"),
        ],
        execute=tool_financial_mark_paid,
        category="financial",
    ))
