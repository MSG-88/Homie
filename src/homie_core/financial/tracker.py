"""Bill tracking — status queries, overdue detection, recurring detection."""
from __future__ import annotations

import time
from typing import Optional

from homie_core.financial.models import BillGroup
from homie_core.vault.models import FinancialRecord


def _parse_amount(amount: str | None) -> float | None:
    """Safely parse amount string to float."""
    if not amount:
        return None
    try:
        return float(amount.replace(",", ""))
    except (ValueError, TypeError):
        return None


class BillTracker:
    def __init__(self, vault):
        self._vault = vault

    def get_pending(self) -> list[FinancialRecord]:
        return self._vault.query_financial(status="pending")

    def get_overdue(self) -> list[FinancialRecord]:
        now = time.time()
        pending = self._vault.query_financial(status="pending")
        return [r for r in pending if r.due_date and r.due_date < now]

    def mark_overdue(self) -> int:
        """Scan pending bills and mark overdue ones. Returns count."""
        overdue = self.get_overdue()
        for r in overdue:
            self._vault.update_financial(r.id, status="overdue")
        return len(overdue)

    def group_by_category(self, records: list[FinancialRecord]) -> dict[str, list[FinancialRecord]]:
        groups: dict[str, list] = {}
        for r in records:
            groups.setdefault(r.category, []).append(r)
        return groups

    def detect_recurring(self, records: list[FinancialRecord]) -> list[BillGroup]:
        """Group records by description similarity and amount proximity (10% tolerance)."""
        groups: dict[str, list[FinancialRecord]] = {}
        for r in records:
            key = r.description.strip().lower()
            groups.setdefault(key, []).append(r)

        result = []
        for desc, recs in groups.items():
            if len(recs) < 2:
                continue
            amounts = [_parse_amount(r.amount) for r in recs]
            valid_amounts = [a for a in amounts if a is not None]
            if not valid_amounts:
                continue
            avg = sum(valid_amounts) / len(valid_amounts)
            # Check all amounts within 10% of average
            if all(abs(a - avg) / max(avg, 0.01) <= 0.1 for a in valid_amounts):
                result.append(BillGroup(
                    description_pattern=recs[0].description,
                    typical_amount=avg,
                    currency=recs[0].currency or "USD",
                    occurrences=len(recs),
                    last_due=max((r.due_date for r in recs if r.due_date), default=None),
                    record_ids=[r.id for r in recs],
                ))
        return result
