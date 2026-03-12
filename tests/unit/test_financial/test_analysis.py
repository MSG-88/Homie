"""Tests for spending analysis."""
from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import MagicMock

from homie_core.financial.analysis import SpendingAnalysis, _parse_amount
from homie_core.vault.models import FinancialRecord


def _make_record(**overrides) -> FinancialRecord:
    defaults = dict(
        id=1, source="gmail:msg1", category="bill",
        description="Test Bill", amount="100.00",
        currency="USD", due_date=None,
        status="pending", reminded_at=None,
        raw_extract=None, created_at=time.time(), updated_at=time.time(),
    )
    defaults.update(overrides)
    return FinancialRecord(**defaults)


class TestMonthlySummary:
    def test_single_month(self):
        now = datetime.now()
        vault = MagicMock()
        vault.query_financial.return_value = [
            _make_record(id=1, amount="100.00", category="bill", created_at=time.time()),
            _make_record(id=2, amount="50.00", category="subscription", created_at=time.time()),
        ]
        analysis = SpendingAnalysis(vault)
        summaries = analysis.monthly_summary(now.year, now.month)
        assert len(summaries) == 1
        assert summaries[0].total_amount == 150.0
        assert summaries[0].by_category["bill"] == 100.0
        assert summaries[0].by_category["subscription"] == 50.0

    def test_multiple_currencies(self):
        now = datetime.now()
        vault = MagicMock()
        vault.query_financial.return_value = [
            _make_record(id=1, amount="100.00", currency="USD", created_at=time.time()),
            _make_record(id=2, amount="80.00", currency="EUR", created_at=time.time()),
        ]
        analysis = SpendingAnalysis(vault)
        summaries = analysis.monthly_summary(now.year, now.month)
        assert len(summaries) == 2
        currencies = {s.currency for s in summaries}
        assert currencies == {"USD", "EUR"}

    def test_none_amount_skipped(self):
        now = datetime.now()
        vault = MagicMock()
        vault.query_financial.return_value = [
            _make_record(id=1, amount=None, created_at=time.time()),
            _make_record(id=2, amount="50.00", created_at=time.time()),
        ]
        analysis = SpendingAnalysis(vault)
        summaries = analysis.monthly_summary(now.year, now.month)
        assert len(summaries) == 1
        assert summaries[0].total_amount == 50.0


class TestTrend:
    def test_trend_stable(self):
        vault = MagicMock()
        vault.query_financial.return_value = []
        analysis = SpendingAnalysis(vault)
        results = analysis.trend()
        assert len(results) == 0  # No data, no trends


class TestLargestBills:
    def test_returns_sorted(self):
        vault = MagicMock()
        vault.query_financial.return_value = [
            _make_record(id=1, amount="50.00"),
            _make_record(id=2, amount="200.00"),
            _make_record(id=3, amount="100.00"),
        ]
        analysis = SpendingAnalysis(vault)
        result = analysis.largest_bills(n=2)
        assert len(result) == 2
        assert result[0].id == 2  # 200.00 first
        assert result[1].id == 3  # 100.00 second
