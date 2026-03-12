"""Bill reminder engine — checks due dates, pushes to WorkingMemory."""
from __future__ import annotations

import time

from homie_core.vault.models import FinancialRecord


class BillReminder:
    DEFAULT_THRESHOLDS = [3 * 86400, 1 * 86400]  # 3 days, 1 day

    def __init__(self, vault, working_memory=None, thresholds: list[int] | None = None):
        self._vault = vault
        self._working_memory = working_memory
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def check(self) -> list[FinancialRecord]:
        """Find pending bills due within the largest threshold window."""
        now = time.time()
        max_window = max(self._thresholds)
        due_before = now + max_window
        pending = self._vault.query_financial(status="pending", due_before=due_before)
        needs_reminder = []
        for r in pending:
            if r.due_date is None:
                continue
            time_until = r.due_date - now
            if time_until < 0:
                continue  # Already overdue, handled by tracker
            for threshold in sorted(self._thresholds):
                if time_until <= threshold:
                    if r.reminded_at and (now - r.reminded_at) < threshold / 2:
                        break  # Already reminded recently for this window
                    needs_reminder.append(r)
                    break
        return needs_reminder

    def remind(self, records: list[FinancialRecord]) -> None:
        now = time.time()
        for r in records:
            self._vault.update_financial(r.id, reminded_at=now)
        if self._working_memory and records:
            summaries = []
            for r in records:
                days_left = (r.due_date - now) / 86400 if r.due_date else 0
                summaries.append({
                    "description": r.description,
                    "amount": r.amount,
                    "currency": r.currency,
                    "days_until_due": round(days_left, 1),
                })
            self._working_memory.update("upcoming_bills", summaries)

    def tick(self) -> str:
        """Called from vault sync manager."""
        upcoming = self.check()
        if upcoming:
            self.remind(upcoming)
            return f"{len(upcoming)} bill reminder(s) sent"
        return "No upcoming bills"
