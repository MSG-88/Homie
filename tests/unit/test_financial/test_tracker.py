"""Tests for bill tracker."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from homie_core.financial.tracker import BillTracker, _parse_amount
from homie_core.vault.models import FinancialRecord


def _make_record(**overrides) -> FinancialRecord:
    defaults = dict(
        id=1, source="gmail:msg1", category="bill",
        description="Electric Bill", amount="142.50",
        currency="USD", due_date=time.time() + 86400,
        status="pending", reminded_at=None,
        raw_extract=None, created_at=time.time(), updated_at=time.time(),
    )
    defaults.update(overrides)
    return FinancialRecord(**defaults)


class TestParseAmount:
    def test_valid(self):
        assert _parse_amount("142.50") == 142.50

    def test_with_commas(self):
        assert _parse_amount("1,250.00") == 1250.0

    def test_none(self):
        assert _parse_amount(None) is None

    def test_invalid(self):
        assert _parse_amount("N/A") is None


class TestPendingBills:
    def test_returns_only_pending(self):
        vault = MagicMock()
        vault.query_financial.return_value = [_make_record()]
        tracker = BillTracker(vault)
        result = tracker.get_pending()
        vault.query_financial.assert_called_with(status="pending")
        assert len(result) == 1


class TestOverdueDetection:
    def test_past_due_detected(self):
        vault = MagicMock()
        vault.query_financial.return_value = [
            _make_record(id=1, due_date=time.time() - 86400),  # past due
            _make_record(id=2, due_date=time.time() + 86400),  # future
        ]
        tracker = BillTracker(vault)
        result = tracker.get_overdue()
        assert len(result) == 1
        assert result[0].id == 1

    def test_mark_overdue_calls_vault(self):
        vault = MagicMock()
        vault.query_financial.return_value = [
            _make_record(id=5, due_date=time.time() - 86400),
        ]
        tracker = BillTracker(vault)
        count = tracker.mark_overdue()
        assert count == 1
        vault.update_financial.assert_called_with(5, status="overdue")


class TestGroupByCategory:
    def test_groups_correctly(self):
        vault = MagicMock()
        tracker = BillTracker(vault)
        records = [
            _make_record(id=1, category="bill"),
            _make_record(id=2, category="subscription"),
            _make_record(id=3, category="bill"),
        ]
        groups = tracker.group_by_category(records)
        assert len(groups["bill"]) == 2
        assert len(groups["subscription"]) == 1


class TestRecurringDetection:
    def test_same_desc_similar_amount_grouped(self):
        vault = MagicMock()
        tracker = BillTracker(vault)
        records = [
            _make_record(id=1, description="Netflix", amount="15.99"),
            _make_record(id=2, description="Netflix", amount="15.99"),
            _make_record(id=3, description="Netflix", amount="16.49"),
        ]
        groups = tracker.detect_recurring(records)
        assert len(groups) == 1
        assert groups[0].occurrences == 3

    def test_different_amounts_not_grouped(self):
        vault = MagicMock()
        tracker = BillTracker(vault)
        records = [
            _make_record(id=1, description="Utility", amount="50.00"),
            _make_record(id=2, description="Utility", amount="200.00"),
        ]
        groups = tracker.detect_recurring(records)
        assert len(groups) == 0  # Amounts differ > 10%
