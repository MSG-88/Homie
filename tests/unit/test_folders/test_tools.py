"""Tests for folder AI tool wrappers."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from homie_core.brain.tool_registry import ToolRegistry
from homie_core.folders.tools import register_folder_tools
from homie_core.folders.models import IndexedFile


def _make_indexed_file(**overrides) -> IndexedFile:
    defaults = dict(
        source="/home/user/docs/report.txt",
        content_type="text/plain",
        summary="Quarterly earnings report",
        topics=["txt", "docs"],
        indexed_at=1700000000.0,
        size=1024,
        modified_at=1699999000.0,
    )
    defaults.update(overrides)
    return IndexedFile(**defaults)


class TestFolderToolRegistration:
    def test_registers_3_tools(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_folder_tools(registry, service)

        tool_names = {t.name for t in registry.list_tools()}
        expected = {"folder_search", "folder_summary", "folder_list_watches"}
        assert expected.issubset(tool_names)

    def test_all_tools_have_folders_category(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_folder_tools(registry, service)

        for tool in registry.list_tools():
            if tool.name.startswith("folder_"):
                assert tool.category == "folders"


class TestFolderSearchTool:
    def test_search_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.search.return_value = [_make_indexed_file()]
        register_folder_tools(registry, service)

        tool = registry.get("folder_search")
        result = tool.execute(query="report")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["source"] == "/home/user/docs/report.txt"
        assert data[0]["content_type"] == "text/plain"

    def test_search_with_max_results_string(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.search.return_value = []
        register_folder_tools(registry, service)

        tool = registry.get("folder_search")
        tool.execute(query="test", max_results="5")
        service.search.assert_called_once_with("test", folder=None, max_results=5)

    def test_search_invalid_max_results_defaults(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.search.return_value = []
        register_folder_tools(registry, service)

        tool = registry.get("folder_search")
        tool.execute(query="test", max_results="abc")
        service.search.assert_called_once_with("test", folder=None, max_results=10)


class TestFolderSummaryTool:
    def test_summary_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_summary.return_value = {
            "watch_count": 2, "total_files": 150, "total_indexed": 120,
        }
        register_folder_tools(registry, service)

        tool = registry.get("folder_summary")
        result = tool.execute()
        data = json.loads(result)
        assert data["watch_count"] == 2
        assert data["total_files"] == 150

    def test_summary_with_folder(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_summary.return_value = {"path": "/docs", "file_count": 50}
        register_folder_tools(registry, service)

        tool = registry.get("folder_summary")
        tool.execute(folder="/docs")
        service.get_summary.assert_called_once_with(folder="/docs")


class TestFolderListWatchesTool:
    def test_list_watches_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.list_watches.return_value = [
            {"id": 1, "path": "/docs", "label": "Documents", "file_count": 42},
        ]
        register_folder_tools(registry, service)

        tool = registry.get("folder_list_watches")
        result = tool.execute()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["label"] == "Documents"
