"""Email organizer — auto-labeling, archiving, and financial extraction.

Works with any EmailProvider to apply labels and archive/trash messages
based on classifier output.
"""
from __future__ import annotations

import re
from typing import Any

from homie_core.email.models import EmailMessage, EmailSyncConfig
from homie_core.email.provider import EmailProvider

# Category → Homie label name mapping
HOMIE_LABELS = {
    "bill": "Homie/Bills",
    "work": "Homie/Work",
    "newsletter": "Homie/Newsletters",
    "social": "Homie/Social",
    "review": "Homie/Review",
    "personal": "Homie/Personal",
    "travel": "Homie/Travel",
    "order": "Homie/Orders",
    "security": "Homie/Security",
    "promotion": "Homie/Promotions",
}

# Currency patterns
_CURRENCY_PATTERNS = [
    (re.compile(r"\$\s*([\d,]+\.?\d*)"), "USD"),
    (re.compile(r"€\s*([\d,]+\.?\d*)"), "EUR"),
    (re.compile(r"£\s*([\d,]+\.?\d*)"), "GBP"),
    (re.compile(r"₹\s*([\d,]+\.?\d*)"), "INR"),
]


class EmailOrganizer:
    """Applies labels, decides archiving, and extracts financial data."""

    def __init__(
        self,
        provider: EmailProvider,
        label_ids: dict[str, str],
    ):
        self._provider = provider
        self._label_ids = label_ids  # category -> Gmail label ID

    def apply_labels(self, msg: EmailMessage) -> None:
        """Apply Homie labels based on message categories."""
        for category in msg.categories:
            label_id = self._label_ids.get(category)
            if label_id:
                self._provider.apply_label(msg.id, label_id)

    def should_archive(self, msg: EmailMessage, user_is_direct: bool = True,
                       sender_open_count: int = 999) -> bool:
        """Decide whether to archive (remove from inbox).

        Args:
            sender_open_count: Number of last N emails from this sender that user opened.
                               Default 999 (don't archive). Pass actual count for newsletters.
        """
        # Social always archived
        if "social" in msg.categories:
            return True
        # Low priority + not direct recipient
        if msg.priority == "low" and not user_is_direct:
            return True
        # Newsletter + user hasn't opened last 3 from same sender (spec Section 8.2)
        if "newsletter" in msg.categories and sender_open_count < 3:
            return True
        return False

    def should_trash(self, msg: EmailMessage, config: EmailSyncConfig) -> bool:
        """Decide whether to trash (spam score > 0.8 and auto_trash enabled)."""
        if not config.auto_trash_spam:
            return False
        return msg.spam_score > 0.8

    def extract_financial(self, msg: EmailMessage) -> dict[str, Any] | None:
        """Extract financial data (amount, currency) from bill emails.

        Returns dict with keys: amount, currency, description, source.
        Returns None if no financial data found.
        """
        text = f"{msg.subject} {msg.snippet}"

        amount = None
        currency = None
        for pattern, curr in _CURRENCY_PATTERNS:
            match = pattern.search(text)
            if match:
                amount = match.group(1).replace(",", "")
                currency = curr
                break

        if amount is None:
            return None

        return {
            "source": f"gmail:{msg.id}",
            "category": "bill",
            "description": msg.subject,
            "amount": amount,
            "currency": currency,
        }
