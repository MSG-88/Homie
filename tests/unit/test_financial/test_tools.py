"""Tests for financial AI tool wrappers."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from homie_core.brain.tool_registry import ToolRegistry
from homie_core.financial.tools import register_financial_tools


class TestFinancialToolRegistration:
    def test_registers_4_tools(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_financial_tools(registry, service)
        tool_names = {t.name for t in registry.list_tools()}
        expected = {"financial_summary", "financial_upcoming", "financial_history", "financial_mark_paid"}
        assert expected.issubset(tool_names)

    def test_all_tools_have_financial_category(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_financial_tools(registry, service)
        for tool in registry.list_tools():
            if tool.name.startswith("financial_"):
                assert tool.category == "financial"


class TestFinancialSummaryTool:
    def test_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_summary.return_value = {"total": 500, "pending_count": 3}
        register_financial_tools(registry, service)
        tool = registry.get("financial_summary")
        result = tool.execute()
        data = json.loads(result)
        assert data["total"] == 500


class TestFinancialUpcomingTool:
    def test_returns_json(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.get_upcoming.return_value = [{"description": "Electric", "amount": "142.50"}]
        register_financial_tools(registry, service)
        tool = registry.get("financial_upcoming")
        result = tool.execute()
        data = json.loads(result)
        assert len(data) == 1


class TestMarkPaidTool:
    def test_returns_status(self):
        registry = ToolRegistry()
        service = MagicMock()
        service.mark_paid.return_value = {"id": 5, "status": "paid"}
        register_financial_tools(registry, service)
        tool = registry.get("financial_mark_paid")
        result = tool.execute(record_id="5")
        data = json.loads(result)
        assert data["status"] == "paid"

    def test_invalid_id_returns_error(self):
        registry = ToolRegistry()
        service = MagicMock()
        register_financial_tools(registry, service)
        tool = registry.get("financial_mark_paid")
        result = tool.execute(record_id="abc")
        data = json.loads(result)
        assert "error" in data
