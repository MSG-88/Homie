"""Email spam scoring, priority scoring, and category detection.

Uses weighted heuristic signals — no external ML dependencies.
Scores are clamped to [0.0, 1.0].
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from homie_core.email.models import EmailMessage

# Spam phrases (lowercase)
_SPAM_PHRASES = {
    "act now", "click here", "limited time", "free gift", "winner",
    "congratulations", "claim your", "unsubscribe", "opt out",
}

# Action keywords for priority (lowercase)
_ACTION_KEYWORDS = {
    "urgent", "deadline", "asap", "payment", "meeting", "rsvp",
    "action required", "due date", "respond", "immediately",
    "important", "critical", "overdue",
}

# Social media senders (partial domain match)
_SOCIAL_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "instagram.com",
    "tiktok.com", "reddit.com", "quora.com", "pinterest.com",
    "facebookmail.com", "x.com",
}

# Bill/invoice patterns
_BILL_PATTERN = re.compile(
    r"(invoice|payment due|amount due|billing|statement|receipt|\$\d+|\€\d+|₹\d+)",
    re.IGNORECASE,
)


def _extract_domain(email_str: str) -> str:
    """Extract domain from 'Name <user@domain>' or 'user@domain'."""
    match = re.search(r"@([\w.-]+)", email_str)
    return match.group(1).lower() if match else ""


def _extract_email(email_str: str) -> str:
    """Extract bare email from 'Name <user@domain>' or 'user@domain'."""
    match = re.search(r"[\w.+-]+@[\w.-]+", email_str)
    return match.group(0).lower() if match else email_str.lower()


class EmailClassifier:
    """Heuristic email classifier for spam, priority, and categories."""

    def __init__(
        self,
        user_email: str,
        reply_history: set[str] | None = None,
        sender_corrections: dict[str, float] | None = None,
    ):
        self._user_email = user_email.lower()
        self._user_domain = _extract_domain(user_email)
        self._reply_history = {e.lower() for e in (reply_history or set())}
        self._sender_corrections = sender_corrections or {}

    def spam_score(
        self,
        msg: EmailMessage,
        headers: dict[str, str] | None = None,
        user_is_direct: bool = True,
    ) -> float:
        """Compute spam score in [0.0, 1.0]. Higher = more likely spam."""
        headers = headers or {}
        score = 0.0
        sender_email = _extract_email(msg.sender)
        sender_domain = _extract_domain(msg.sender)

        # Positive signals (increase spam likelihood)
        if sender_email not in self._reply_history:
            score += 0.3

        if headers.get("List-Unsubscribe"):
            score += 0.2

        if headers.get("Precedence", "").lower() == "bulk":
            score += 0.2

        # Subject analysis
        subject_upper_ratio = sum(1 for c in msg.subject if c.isupper()) / max(len(msg.subject), 1)
        if subject_upper_ratio > 0.6 and len(msg.subject) > 5:
            score += 0.2

        excessive_punct = len(re.findall(r"[!?]{2,}", msg.subject))
        if excessive_punct > 0:
            score += 0.1

        text = (msg.subject + " " + msg.snippet).lower()
        spam_hit = any(phrase in text for phrase in _SPAM_PHRASES)
        if spam_hit:
            score += 0.1

        # Negative signals (decrease spam likelihood)
        if sender_email in self._reply_history:
            score -= 0.5

        if user_is_direct:
            score -= 0.2

        if sender_domain == self._user_domain and self._user_domain:
            score -= 0.3

        # Per-sender correction from learning
        if sender_email in self._sender_corrections:
            score += self._sender_corrections[sender_email]

        return max(0.0, min(1.0, score))

    def priority_score(
        self,
        msg: EmailMessage,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Classify priority as 'high', 'medium', or 'low'."""
        headers = headers or {}
        sender_email = _extract_email(msg.sender)
        sender_domain = _extract_domain(msg.sender)
        text = (msg.subject + " " + msg.snippet).lower()

        is_known = sender_email in self._reply_history
        has_action = any(kw in text for kw in _ACTION_KEYWORDS)
        is_list = bool(headers.get("List-Unsubscribe"))
        is_social = any(d in sender_domain for d in _SOCIAL_DOMAINS)

        # High: known contact + action words, or same domain + action
        if (is_known and has_action) or (sender_domain == self._user_domain and has_action):
            return "high"

        # Low: mailing lists, social, automated
        if is_list or is_social:
            return "low"

        # Medium: everything else (direct unknown sender, etc.)
        return "medium"

    def detect_categories(
        self,
        msg: EmailMessage,
        headers: dict[str, str] | None = None,
    ) -> list[str]:
        """Detect content categories for auto-labeling."""
        headers = headers or {}
        categories = []
        sender_domain = _extract_domain(msg.sender)
        text = (msg.subject + " " + (msg.snippet or "")).lower()

        # Bill detection
        if _BILL_PATTERN.search(text):
            categories.append("bill")

        # Work: same domain
        if sender_domain == self._user_domain and self._user_domain:
            categories.append("work")

        # Newsletter
        if headers.get("List-Unsubscribe") and not any(d in sender_domain for d in _SOCIAL_DOMAINS):
            categories.append("newsletter")

        # Social
        if any(d in sender_domain for d in _SOCIAL_DOMAINS):
            categories.append("social")

        return categories
