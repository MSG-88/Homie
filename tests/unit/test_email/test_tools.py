"""Tests for email AI tool wrappers."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from homie_core.brain.tool_registry import ToolRegistry
from homie_core.email.tools import register_email_tools
from homie_core.email.models import EmailMessage


def _make_msg(**overrides) -> EmailMessage:
    defaults = dict(
        id="msg1", thread_id="t1", account_id="user@gmail.com",
        provider="gmail", subject="Hello", sender="alice@x.com",
        recipients=["user@gmail.com"], snippet="Hey there...",
        priority="high", spam_score=0.0, is_read=False,
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


class TestEmailToolRegistration:
    def test_registers_9_tools(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        register_email_tools(registry, email_service)

        tool_names = {t.name for t in registry.list_tools()}
        expected = {
            "email_search", "email_read", "email_thread",
            "email_draft", "email_labels", "email_summary",
            "email_unread", "email_archive", "email_mark_read",
        }
        assert expected.issubset(tool_names)

    def test_all_tools_have_email_category(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        register_email_tools(registry, email_service)

        for tool in registry.list_tools():
            if tool.name.startswith("email_"):
                assert tool.category == "email"


class TestEmailSearchTool:
    def test_search_returns_json(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        email_service.search.return_value = [_make_msg()]
        register_email_tools(registry, email_service)

        tool = registry.get("email_search")
        result = tool.execute(query="from:alice")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["subject"] == "Hello"


class TestEmailReadTool:
    def test_read_returns_body(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        email_service.read_message.return_value = {
            "subject": "Hello",
            "sender": "alice@x.com",
            "body": "Full message body here",
        }
        register_email_tools(registry, email_service)

        tool = registry.get("email_read")
        result = tool.execute(message_id="msg1")
        data = json.loads(result)
        assert data["body"] == "Full message body here"


class TestEmailDraftTool:
    def test_draft_returns_id(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        email_service.create_draft.return_value = "draft_123"
        register_email_tools(registry, email_service)

        tool = registry.get("email_draft")
        result = tool.execute(to="bob@x.com", subject="Re: Hi", body="Thanks!")
        data = json.loads(result)
        assert data["draft_id"] == "draft_123"


class TestEmailUnreadTool:
    def test_unread_returns_grouped(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        email_service.get_unread.return_value = {
            "high": [_make_msg().to_dict()],
            "medium": [],
            "low": [],
        }
        register_email_tools(registry, email_service)

        tool = registry.get("email_unread")
        result = tool.execute()
        data = json.loads(result)
        assert "high" in data
