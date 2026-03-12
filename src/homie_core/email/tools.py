"""AI tool wrappers for email integration.

Registers 9 tools with the ToolRegistry so the Brain can interact
with email via the agentic loop.
"""
from __future__ import annotations

import json

from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "..."
    return text


def register_email_tools(registry: ToolRegistry, email_service) -> None:
    """Register all email tools with the tool registry."""

    def tool_email_search(query: str, account: str = "all", max_results: str = "10") -> str:
        results = email_service.search(query, account=account, max_results=int(max_results))
        return _truncate(json.dumps([m.to_dict() if hasattr(m, "to_dict") else m for m in results]))

    registry.register(Tool(
        name="email_search",
        description="Search emails using Gmail query syntax (e.g. 'from:alice subject:meeting').",
        params=[
            ToolParam(name="query", description="Gmail search query", type="string"),
            ToolParam(name="account", description="Account email or 'all'", type="string", required=False, default="all"),
            ToolParam(name="max_results", description="Maximum results", type="string", required=False, default="10"),
        ],
        execute=tool_email_search,
        category="email",
    ))

    def tool_email_read(message_id: str) -> str:
        result = email_service.read_message(message_id)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_read",
        description="Read the full body of a specific email by its message ID.",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_read,
        category="email",
    ))

    def tool_email_thread(thread_id: str) -> str:
        result = email_service.get_thread(thread_id)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_thread",
        description="Get all messages in a conversation thread.",
        params=[
            ToolParam(name="thread_id", description="Thread/conversation ID", type="string"),
        ],
        execute=tool_email_thread,
        category="email",
    ))

    def tool_email_draft(
        to: str, subject: str, body: str,
        reply_to: str = "", cc: str = "", bcc: str = "",
        account: str = "",
    ) -> str:
        draft_id = email_service.create_draft(
            to=to, subject=subject, body=body,
            reply_to=reply_to or None,
            cc=[c.strip() for c in cc.split(",") if c.strip()] if cc else None,
            bcc=[b.strip() for b in bcc.split(",") if b.strip()] if bcc else None,
            account=account or None,
        )
        return json.dumps({"draft_id": draft_id, "status": "Draft created. User must review and send manually."})

    registry.register(Tool(
        name="email_draft",
        description="Create a draft email (never sends automatically). User reviews and sends manually.",
        params=[
            ToolParam(name="to", description="Recipient email", type="string"),
            ToolParam(name="subject", description="Email subject", type="string"),
            ToolParam(name="body", description="Email body text", type="string"),
            ToolParam(name="reply_to", description="Message ID to reply to", type="string", required=False, default=""),
            ToolParam(name="cc", description="CC recipients (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="bcc", description="BCC recipients (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="account", description="Send from this account", type="string", required=False, default=""),
        ],
        execute=tool_email_draft,
        category="email",
    ))

    def tool_email_labels(account: str = "") -> str:
        labels = email_service.list_labels(account=account or None)
        return json.dumps(labels)

    registry.register(Tool(
        name="email_labels",
        description="List all email labels/folders.",
        params=[
            ToolParam(name="account", description="Account email (optional)", type="string", required=False, default=""),
        ],
        execute=tool_email_labels,
        category="email",
    ))

    def tool_email_summary(days: str = "1") -> str:
        result = email_service.get_summary(days=int(days))
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_summary",
        description="Get email summary: unread count, high-priority items, action items.",
        params=[
            ToolParam(name="days", description="Number of days to summarize", type="string", required=False, default="1"),
        ],
        execute=tool_email_summary,
        category="email",
    ))

    def tool_email_unread(account: str = "all") -> str:
        result = email_service.get_unread(account=account)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_unread",
        description="List unread emails grouped by priority (high, medium, low).",
        params=[
            ToolParam(name="account", description="Account email or 'all'", type="string", required=False, default="all"),
        ],
        execute=tool_email_unread,
        category="email",
    ))

    def tool_email_archive(message_id: str) -> str:
        email_service.archive_message(message_id)
        return json.dumps({"status": "archived", "message_id": message_id})

    registry.register(Tool(
        name="email_archive",
        description="Archive a message (remove from inbox).",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_archive,
        category="email",
    ))

    def tool_email_mark_read(message_id: str) -> str:
        email_service.mark_read(message_id)
        return json.dumps({"status": "marked_read", "message_id": message_id})

    registry.register(Tool(
        name="email_mark_read",
        description="Mark a message as read.",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_mark_read,
        category="email",
    ))
