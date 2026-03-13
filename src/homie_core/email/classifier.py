"""Email spam scoring, priority scoring, and category detection.

Two-tier classification:
1. Fast heuristic pass (keywords, patterns, domain checks)
2. LLM-powered analysis for ambiguous emails — uses the local model
   to understand actual content and intent.

Scores are clamped to [0.0, 1.0].
"""
from __future__ import annotations

import json
import logging
import re

from homie_core.email.models import EmailMessage

_log = logging.getLogger(__name__)

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

# Social media senders (suffix domain match)
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

# ── LLM classification prompt ────────────────────────────────────────────────

_LLM_CLASSIFY_PROMPT = """\
You are an email triage assistant. Analyze the email below and return ONLY a JSON object — no extra text.

Email:
From: {sender}
To: {recipients}
Subject: {subject}
Date: {date}
Body:
{body}

Return this exact JSON structure:
{{"spam_score": <float 0.0-1.0>, "priority": "<high|medium|low>", "categories": [<list from: bill, work, newsletter, social, personal, travel, order, security, promotion>], "intent": "<one sentence: what does this email want from the reader?>", "action_needed": <true|false>, "summary": "<2-3 sentence summary of content>"}}

Rules:
- spam_score: 0.0 = definitely legitimate, 1.0 = definitely spam/phishing
- priority "high" = needs action soon, "medium" = informational but relevant, "low" = noise/promotional
- categories: pick ALL that apply
- intent: what the sender wants (e.g. "Wants you to pay invoice #1234", "Marketing promotion, no action needed")
- action_needed: true if the reader must do something
- summary: brief factual summary of the email content"""

_LLM_TRIAGE_PROMPT = """\
You are an email triage assistant. Analyze these {count} emails and return ONLY a JSON array — no extra text.

{emails_block}

For each email return:
{{"id": "<email_id>", "spam_score": <0.0-1.0>, "priority": "<high|medium|low>", "categories": [<list>], "intent": "<one sentence>", "action_needed": <true|false>, "summary": "<2-3 sentences>"}}

Category options: bill, work, newsletter, social, personal, travel, order, security, promotion
Return a JSON array of objects, one per email."""

_LLM_DIGEST_PROMPT = """\
You are a personal email assistant. Given these emails from the last {days} day(s), create a brief digest for your user.

{emails_block}

Write a concise digest with these sections:
1. **Action Required** — emails that need a response or action, with sender and what's needed
2. **Important Updates** — notable emails that don't need action but are worth knowing
3. **Noise** — count of promotional, social notifications, newsletters (don't list each one)

Keep it brief and scannable. Use bullet points. Skip empty sections."""

_LLM_DEEP_ANALYZE_PROMPT = """\
You are a proactive personal email assistant. Deeply analyze this email and determine what your user needs to do.

Email:
From: {sender}
To: {recipients}
Subject: {subject}
Date: {date}
Body:
{body}

Return ONLY a JSON object:
{{"urgency": "<immediate|today|this_week|no_action>", "deadline": "<extracted deadline date or null>", "action_type": "<reply|pay|attend|review|sign|confirm|register|cancel|none>", "action_detail": "<specific action: e.g. 'Pay AWS bill of $X before account suspension', 'Confirm flight booking Z9BETM for Mar 15'>", "context": "<why this matters — business impact, financial consequence, or personal relevance>", "suggested_response": "<draft reply text if a reply is needed, or null>", "followup_reminder": "<when to remind user if they haven't acted, e.g. 'tomorrow 9am', or null>"}}"

Rules:
- urgency "immediate": payment overdue, account suspension, security alerts, flights today
- urgency "today": meetings today, deadlines today, time-sensitive offers
- urgency "this_week": bills due this week, upcoming travel, pending reviews
- action_detail must be specific and actionable — not generic
- context explains business/personal impact if user doesn't act
- suggested_response: only for emails that clearly need a reply"""


def _is_social_domain(domain: str) -> bool:
    """Check if domain matches a known social media domain (suffix match)."""
    return any(
        domain == d or domain.endswith("." + d)
        for d in _SOCIAL_DOMAINS
    )


def _extract_domain(email_str: str) -> str:
    """Extract domain from 'Name <user@domain>' or 'user@domain'."""
    match = re.search(r"@([\w.-]+)", email_str)
    return match.group(1).lower() if match else ""


def _extract_email(email_str: str) -> str:
    """Extract bare email from 'Name <user@domain>' or 'user@domain'."""
    match = re.search(r"[\w.+-]+@[\w.-]+", email_str)
    return match.group(0).lower() if match else email_str.lower()


def clean_snippet(text: str) -> str:
    """Strip HTML entities, invisible Unicode chars, and whitespace padding."""
    if not text:
        return ""
    import html as _html
    # Decode HTML entities (&#39; -> ', &amp; -> &, etc.)
    text = _html.unescape(text)
    # Remove invisible Unicode: zero-width spaces, joiners, combining chars, etc.
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\u034f\ufeff\u00ad]+", "", text)
    # Collapse runs of whitespace
    text = re.sub(r"\s{3,}", " ", text).strip()
    return text


def _email_text(msg: EmailMessage, max_body: int = 1500) -> str:
    """Build the text representation of an email for LLM analysis."""
    body = clean_snippet((msg.body or msg.snippet or "")[:max_body])
    return body


def _parse_llm_json(raw: str) -> dict | list | None:
    """Extract JSON from LLM output, handling markdown fences and preamble."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    # Find first { or [
    for i, c in enumerate(cleaned):
        if c in "{[":
            cleaned = cleaned[i:]
            break
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find balanced braces/brackets
        depth = 0
        start = None
        opener = cleaned[0] if cleaned else ""
        closer = "}" if opener == "{" else "]" if opener == "[" else ""
        if not closer:
            return None
        for i, c in enumerate(cleaned):
            if c == opener:
                if start is None:
                    start = i
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


class EmailClassifier:
    """Two-tier email classifier: fast heuristics + optional LLM analysis."""

    def __init__(
        self,
        user_email: str,
        reply_history: set[str] | None = None,
        sender_corrections: dict[str, float] | None = None,
        model_engine=None,
    ):
        self._user_email = user_email.lower()
        self._user_domain = _extract_domain(user_email)
        self._reply_history = {e.lower() for e in (reply_history or set())}
        self._sender_corrections = sender_corrections or {}
        self._model = model_engine  # ModelEngine instance or None

    @property
    def has_llm(self) -> bool:
        return self._model is not None and self._model.is_loaded

    # ── Heuristic methods (unchanged, always available) ───────────────────

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

        text = (msg.subject + " " + (msg.snippet or "")).lower()
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
        text = (msg.subject + " " + (msg.snippet or "")).lower()

        is_known = sender_email in self._reply_history
        has_action = any(kw in text for kw in _ACTION_KEYWORDS)
        is_list = bool(headers.get("List-Unsubscribe"))
        is_social = _is_social_domain(sender_domain)

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
        if headers.get("List-Unsubscribe") and not _is_social_domain(sender_domain):
            categories.append("newsletter")

        # Social
        if _is_social_domain(sender_domain):
            categories.append("social")

        return categories

    # ── LLM-powered methods ──────────────────────────────────────────────

    def llm_classify(self, msg: EmailMessage) -> dict | None:
        """Use the local LLM to deeply analyze a single email.

        Returns dict with: spam_score, priority, categories, intent,
        action_needed, summary.  Returns None if LLM unavailable.
        """
        if not self.has_llm:
            return None

        from datetime import datetime
        date_str = datetime.fromtimestamp(msg.date).strftime("%Y-%m-%d %H:%M") if msg.date else "unknown"
        prompt = _LLM_CLASSIFY_PROMPT.format(
            sender=msg.sender,
            recipients=", ".join(msg.recipients[:5]),
            subject=msg.subject,
            date=date_str,
            body=_email_text(msg),
        )

        try:
            raw = self._model.generate(prompt, max_tokens=512, temperature=0.1, timeout=30)
            result = _parse_llm_json(raw)
            if isinstance(result, dict) and "spam_score" in result:
                # Clamp and validate
                result["spam_score"] = max(0.0, min(1.0, float(result.get("spam_score", 0.5))))
                if result.get("priority") not in ("high", "medium", "low"):
                    result["priority"] = "medium"
                if not isinstance(result.get("categories"), list):
                    result["categories"] = []
                result["action_needed"] = bool(result.get("action_needed", False))
                return result
        except Exception as e:
            _log.debug("LLM classify failed: %s", e)

        return None

    def llm_triage_batch(self, messages: list[EmailMessage]) -> list[dict] | None:
        """Analyze a batch of emails in one LLM call (up to ~15 emails).

        Returns list of dicts with same structure as llm_classify, or None.
        """
        if not self.has_llm or not messages:
            return None

        from datetime import datetime

        # Build compact email block
        parts = []
        for i, msg in enumerate(messages[:15]):
            date_str = datetime.fromtimestamp(msg.date).strftime("%Y-%m-%d %H:%M") if msg.date else "unknown"
            body = _email_text(msg, max_body=500)
            parts.append(
                f"--- Email {i+1} (id: {msg.id}) ---\n"
                f"From: {msg.sender}\nSubject: {msg.subject}\nDate: {date_str}\n"
                f"Body: {body}\n"
            )

        prompt = _LLM_TRIAGE_PROMPT.format(
            count=len(messages[:15]),
            emails_block="\n".join(parts),
        )

        try:
            raw = self._model.generate(prompt, max_tokens=1024, temperature=0.1, timeout=60)
            result = _parse_llm_json(raw)
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        item["spam_score"] = max(0.0, min(1.0, float(item.get("spam_score", 0.5))))
                        if item.get("priority") not in ("high", "medium", "low"):
                            item["priority"] = "medium"
                        item["action_needed"] = bool(item.get("action_needed", False))
                return result
        except Exception as e:
            _log.debug("LLM batch triage failed: %s", e)

        return None

    def llm_digest(self, messages: list[EmailMessage], days: int = 1) -> str | None:
        """Generate a natural-language email digest using the LLM.

        Returns markdown-formatted digest string, or None if LLM unavailable.
        """
        if not self.has_llm or not messages:
            return None

        from datetime import datetime

        parts = []
        for msg in messages[:30]:
            date_str = datetime.fromtimestamp(msg.date).strftime("%Y-%m-%d %H:%M") if msg.date else ""
            parts.append(
                f"- From: {msg.sender} | Subject: {msg.subject} | "
                f"Priority: {msg.priority} | Spam: {msg.spam_score:.1f} | "
                f"Date: {date_str} | Snippet: {(msg.snippet or '')[:200]}"
            )

        prompt = _LLM_DIGEST_PROMPT.format(
            days=days,
            emails_block="\n".join(parts),
        )

        try:
            return self._model.generate(prompt, max_tokens=768, temperature=0.3, timeout=45)
        except Exception as e:
            _log.debug("LLM digest failed: %s", e)

        return None

    def llm_deep_analyze(self, msg: EmailMessage) -> dict | None:
        """Deep contextual analysis: extract deadlines, required actions,
        business impact, and draft responses for actionable emails.

        Returns dict with: urgency, deadline, action_type, action_detail,
        context, suggested_response, followup_reminder. Or None if unavailable.
        """
        if not self.has_llm:
            return None

        from datetime import datetime
        date_str = datetime.fromtimestamp(msg.date).strftime("%Y-%m-%d %H:%M") if msg.date else "unknown"
        prompt = _LLM_DEEP_ANALYZE_PROMPT.format(
            sender=msg.sender,
            recipients=", ".join(msg.recipients[:5]),
            subject=msg.subject,
            date=date_str,
            body=_email_text(msg, max_body=2000),
        )

        try:
            raw = self._model.generate(prompt, max_tokens=512, temperature=0.1, timeout=45)
            result = _parse_llm_json(raw)
            if isinstance(result, dict) and "urgency" in result:
                # Validate urgency
                if result.get("urgency") not in ("immediate", "today", "this_week", "no_action"):
                    result["urgency"] = "no_action"
                return result
        except Exception as e:
            _log.debug("LLM deep analyze failed: %s", e)

        return None
