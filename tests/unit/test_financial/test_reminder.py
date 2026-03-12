"""Tests for bill reminder engine."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from homie_core.financial.reminder import BillReminder
from homie_core.vault.models import FinancialRecord


def _make_record(**overrides) -> FinancialRecord:
    defaults = dict(
        id=1, source="gmail:msg1", category="bill",
        description="Electric Bill", amount="142.50",
        currency="USD", due_date=time.time() + 2 * 86400,
        status="pending", reminded_at=None,
        raw_extract=None, created_at=time.time(), updated_at=time.time(),
    )
    defaults.update(overrides)
    return FinancialRecord(**defaults)


class TestReminderCheck:
    def test_finds_upcoming_within_threshold(self):
        vault = MagicMock()
        vault.query_financial.return_value = [
            _make_record(due_date=time.time() + 2 * 86400),  # 2 days out
        ]
        reminder = BillReminder(vault, thresholds=[3 * 86400])
        result = reminder.check()
        assert len(result) == 1

    def test_skips_already_reminded(self):
        vault = MagicMock()
        now = time.time()
        vault.query_financial.return_value = [
            _make_record(due_date=now + 2 * 86400, reminded_at=now - 3600),  # reminded 1h ago
        ]
        reminder = BillReminder(vault, thresholds=[3 * 86400])
        result = reminder.check()
        assert len(result) == 0

    def test_skips_overdue(self):
        vault = MagicMock()
        vault.query_financial.return_value = [
            _make_record(due_date=time.time() - 86400),  # past due
        ]
        reminder = BillReminder(vault, thresholds=[3 * 86400])
        result = reminder.check()
        assert len(result) == 0


class TestRemind:
    def test_updates_reminded_at(self):
        vault = MagicMock()
        records = [_make_record(id=7)]
        reminder = BillReminder(vault)
        reminder.remind(records)
        vault.update_financial.assert_called_once()
        call_args = vault.update_financial.call_args
        assert call_args[0][0] == 7
        assert "reminded_at" in call_args[1]

    def test_pushes_to_working_memory(self):
        vault = MagicMock()
        wm = MagicMock()
        records = [_make_record(id=7)]
        reminder = BillReminder(vault, working_memory=wm)
        reminder.remind(records)
        wm.update.assert_called_once()
        assert wm.update.call_args[0][0] == "upcoming_bills"


class TestTick:
    def test_tick_returns_status(self):
        vault = MagicMock()
        vault.query_financial.return_value = []
        reminder = BillReminder(vault)
        result = reminder.tick()
        assert result == "No upcoming bills"
