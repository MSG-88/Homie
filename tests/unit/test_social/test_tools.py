"""Tests for social AI tool registration and execution."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from homie_core.brain.tool_registry import ToolRegistry
from homie_core.social.tools import register_social_tools


class TestSocialToolRegistration:
    def test_registers_4_tools(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        register_social_tools(registry, social_service)

        tool_names = {t.name for t in registry.list_tools()}
        expected = {"social_channels", "social_messages", "social_search", "social_unread"}
        assert expected.issubset(tool_names)

    def test_all_tools_have_social_category(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        register_social_tools(registry, social_service)

        for tool in registry.list_tools():
            if tool.name.startswith("social_"):
                assert tool.category == "social"


class TestSocialChannelsTool:
    def test_channels_returns_json(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        social_service.list_channels.return_value = [
            {"id": "C001", "name": "general", "platform": "slack", "is_dm": False, "member_count": 50},
        ]
        register_social_tools(registry, social_service)

        tool = registry.get("social_channels")
        result = tool.execute()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["name"] == "general"

    def test_channels_passes_platform(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        social_service.list_channels.return_value = []
        register_social_tools(registry, social_service)

        tool = registry.get("social_channels")
        tool.execute(platform="slack")
        social_service.list_channels.assert_called_with(platform="slack")


class TestSocialMessagesTool:
    def test_messages_returns_json(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        social_service.get_messages.return_value = [
            {"id": "msg1", "content": "Hello!", "sender": "U123"},
        ]
        register_social_tools(registry, social_service)

        tool = registry.get("social_messages")
        result = tool.execute(channel_id="C001")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["content"] == "Hello!"

    def test_messages_int_guard_on_limit(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        social_service.get_messages.return_value = []
        register_social_tools(registry, social_service)

        tool = registry.get("social_messages")
        tool.execute(channel_id="C001", limit="not_a_number")
        social_service.get_messages.assert_called_with(
            "C001", platform=None, limit=20,
        )


class TestSocialSearchTool:
    def test_search_returns_json(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        social_service.search.return_value = [
            {"id": "msg1", "content": "Found it"},
        ]
        register_social_tools(registry, social_service)

        tool = registry.get("social_search")
        result = tool.execute(query="project update")
        data = json.loads(result)
        assert len(data) == 1

    def test_search_int_guard_on_limit(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        social_service.search.return_value = []
        register_social_tools(registry, social_service)

        tool = registry.get("social_search")
        tool.execute(query="test", limit="bad")
        social_service.search.assert_called_with("test", platform="all", limit=10)


class TestSocialUnreadTool:
    def test_unread_returns_json(self):
        registry = ToolRegistry()
        social_service = MagicMock()
        social_service.get_unread.return_value = {
            "slack": [{"id": "msg1", "content": "Hey!"}],
        }
        register_social_tools(registry, social_service)

        tool = registry.get("social_unread")
        result = tool.execute()
        data = json.loads(result)
        assert "slack" in data
        assert len(data["slack"]) == 1
