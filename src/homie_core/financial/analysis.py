"""Spending analysis — monthly summaries, trends, largest bills."""
from __future__ import annotations

import calendar
import time
from datetime import datetime

from homie_core.financial.models import SpendingSummary, TrendResult
from homie_core.vault.models import FinancialRecord


def _parse_amount(amount: str | None) -> float | None:
    if not amount:
        return None
    try:
        return float(amount.replace(",", ""))
    except (ValueError, TypeError):
        return None


class SpendingAnalysis:
    def __init__(self, vault):
        self._vault = vault

    def monthly_summary(self, year: int, month: int) -> list[SpendingSummary]:
        """Compute spending for a given month. Returns one SpendingSummary per currency."""
        start = datetime(year, month, 1).timestamp()
        _, last_day = calendar.monthrange(year, month)
        end = datetime(year, month, last_day, 23, 59, 59).timestamp()

        all_records = self._vault.query_financial()
        records = [r for r in all_records if r.created_at and start <= r.created_at <= end]

        by_currency: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        for r in records:
            amt = _parse_amount(r.amount)
            if amt is None:
                continue
            cur = r.currency or "USD"
            if cur not in by_currency:
                by_currency[cur] = {}
                counts[cur] = 0
            by_currency[cur].setdefault(r.category, 0.0)
            by_currency[cur][r.category] += amt
            counts[cur] += 1

        summaries = []
        for cur, cats in by_currency.items():
            summaries.append(SpendingSummary(
                period=f"{year}-{month:02d}",
                total_amount=sum(cats.values()),
                currency=cur,
                by_category=cats,
                record_count=counts[cur],
            ))
        return summaries

    def trend(self, months: int = 2) -> list[TrendResult]:
        """Compare current month vs previous. One TrendResult per currency."""
        now = datetime.now()
        curr_year, curr_month = now.year, now.month
        prev_month = curr_month - 1 if curr_month > 1 else 12
        prev_year = curr_year if curr_month > 1 else curr_year - 1

        current = {s.currency: s.total_amount for s in self.monthly_summary(curr_year, curr_month)}
        previous = {s.currency: s.total_amount for s in self.monthly_summary(prev_year, prev_month)}

        results = []
        all_currencies = set(current) | set(previous)
        for cur in all_currencies:
            curr_amt = current.get(cur, 0.0)
            prev_amt = previous.get(cur, 0.0)
            if prev_amt == 0 and curr_amt == 0:
                continue
            if prev_amt == 0:
                change_pct = 100.0
            else:
                change_pct = ((curr_amt - prev_amt) / prev_amt) * 100

            if abs(change_pct) < 5:
                direction = "stable"
            elif change_pct > 0:
                direction = "up"
            else:
                direction = "down"

            results.append(TrendResult(
                direction=direction,
                current_month=curr_amt,
                previous_month=prev_amt,
                change_pct=round(change_pct, 1),
                currency=cur,
            ))
        return results

    def largest_bills(self, n: int = 5) -> list[FinancialRecord]:
        """Return top N records by amount descending."""
        all_records = self._vault.query_financial()
        with_amounts = []
        for r in all_records:
            amt = _parse_amount(r.amount)
            if amt is not None:
                with_amounts.append((r, amt))
        with_amounts.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in with_amounts[:n]]
