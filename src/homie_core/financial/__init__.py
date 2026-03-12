"""Financial intelligence — bill tracking, reminders, spending analysis.

FinancialService is the main facade used by the daemon and CLI.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from homie_core.vault.models import FinancialRecord


class FinancialService:
    """High-level facade for financial operations."""

    def __init__(self, vault, working_memory=None):
        from homie_core.financial.tracker import BillTracker
        from homie_core.financial.reminder import BillReminder
        from homie_core.financial.analysis import SpendingAnalysis

        self._vault = vault
        self._tracker = BillTracker(vault)
        self._reminder = BillReminder(vault, working_memory=working_memory)
        self._analysis = SpendingAnalysis(vault)

    def get_summary(self, year: int | None = None, month: int | None = None) -> dict:
        """Get spending summary for a month (defaults to current)."""
        from datetime import datetime
        now = datetime.now()
        y = year or now.year
        m = month or now.month
        summaries = self._analysis.monthly_summary(y, m)
        pending = self._vault.query_financial(status="pending")
        overdue = self._tracker.get_overdue()
        return {
            "year": y, "month": m,
            "summaries": [s.__dict__ for s in summaries],
            "pending_count": len(pending),
            "overdue_count": len(overdue),
        }

    def get_upcoming(self, days: int = 7) -> list[dict]:
        """List bills due within N days."""
        due_before = time.time() + (days * 86400)
        records = self._vault.query_financial(status="pending", due_before=due_before)
        return [_record_to_dict(r) for r in records]

    def get_history(self, category: str | None = None, limit: int = 20) -> list[dict]:
        """Query past financial records."""
        records = self._vault.query_financial(category=category)
        return [_record_to_dict(r) for r in records[:limit]]

    def mark_paid(self, record_id: int) -> dict:
        """Mark a bill as paid."""
        self._vault.update_financial(record_id, status="paid")
        return {"id": record_id, "status": "paid"}

    def reminder_tick(self) -> str:
        """Called from vault sync manager."""
        overdue_count = self._tracker.mark_overdue()
        reminder_result = self._reminder.tick()
        parts = []
        if overdue_count:
            parts.append(f"{overdue_count} marked overdue")
        parts.append(reminder_result)
        return "; ".join(parts)


def _record_to_dict(r: FinancialRecord) -> dict:
    return {
        "id": r.id, "source": r.source, "category": r.category,
        "description": r.description, "amount": r.amount,
        "currency": r.currency, "due_date": r.due_date,
        "status": r.status, "reminded_at": r.reminded_at,
    }
