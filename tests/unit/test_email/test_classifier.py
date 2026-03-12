"""Tests for email spam and priority classifier."""
from __future__ import annotations

from homie_core.email.classifier import EmailClassifier
from homie_core.email.models import EmailMessage


def _make_msg(**overrides) -> EmailMessage:
    """Helper to create test EmailMessage with defaults."""
    defaults = dict(
        id="msg1", thread_id="t1", account_id="user@work.com",
        provider="gmail", subject="Hello", sender="alice@example.com",
        recipients=["user@work.com"], snippet="Hey there...",
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


class TestSpamScoring:
    def test_clean_email_from_known_contact(self):
        classifier = EmailClassifier(
            user_email="user@work.com",
            reply_history={"alice@example.com"},
        )
        msg = _make_msg(sender="alice@example.com")
        score = classifier.spam_score(msg)
        assert score < 0.3, f"Known contact should score low, got {score}"

    def test_bulk_sender_scores_higher(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            sender="noreply@marketing.com",
            subject="AMAZING DEAL!!!",
            snippet="Unsubscribe from this list",
        )
        # Simulate headers via labels (bulk indicator)
        score = classifier.spam_score(msg, headers={"Precedence": "bulk", "List-Unsubscribe": "yes"})
        assert score > 0.3, f"Bulk sender should score higher, got {score}"

    def test_same_domain_reduces_score(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(sender="boss@work.com")
        score = classifier.spam_score(msg)
        assert score < 0.3, f"Same domain should score low, got {score}"

    def test_score_clamped_to_0_1(self):
        classifier = EmailClassifier(
            user_email="user@work.com",
            reply_history={"sender@x.com"},
        )
        # Known contact with same domain — many negative signals
        msg = _make_msg(sender="sender@work.com")
        score = classifier.spam_score(msg)
        assert 0.0 <= score <= 1.0

    def test_all_caps_subject_increases_score(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(subject="FREE MONEY NOW ACT FAST")
        score = classifier.spam_score(msg)
        normal_msg = _make_msg(subject="Meeting tomorrow")
        normal_score = classifier.spam_score(normal_msg)
        assert score > normal_score

    def test_cc_recipient_increases_score(self):
        classifier = EmailClassifier(user_email="user@work.com")
        # User is in CC, not direct To
        msg = _make_msg(recipients=["other@work.com", "user@work.com"])
        # Simulate: user is not the primary To recipient
        score_cc = classifier.spam_score(msg, user_is_direct=False)
        score_direct = classifier.spam_score(msg, user_is_direct=True)
        assert score_cc >= score_direct


class TestPriorityScoring:
    def test_known_contact_with_action_words_is_high(self):
        classifier = EmailClassifier(
            user_email="user@work.com",
            reply_history={"boss@work.com"},
        )
        msg = _make_msg(
            sender="boss@work.com",
            subject="Deadline moved to Friday — urgent",
        )
        priority = classifier.priority_score(msg)
        assert priority == "high"

    def test_unknown_direct_sender_is_medium(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(sender="newperson@other.com")
        priority = classifier.priority_score(msg)
        assert priority == "medium"

    def test_mailing_list_is_low(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            sender="noreply@newsletter.com",
            subject="Weekly digest",
        )
        priority = classifier.priority_score(msg, headers={"List-Unsubscribe": "yes"})
        assert priority == "low"


class TestCategoryDetection:
    def test_bill_detected(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            subject="Invoice #4521 — Payment Due March 20",
            snippet="Amount due: $142.50",
        )
        categories = classifier.detect_categories(msg)
        assert "bill" in categories

    def test_newsletter_detected(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            sender="news@techblog.com",
            subject="This Week in Tech — Issue #312",
        )
        categories = classifier.detect_categories(msg, headers={"List-Unsubscribe": "yes"})
        assert "newsletter" in categories

    def test_social_detected(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            sender="notifications@linkedin.com",
            subject="John viewed your profile",
        )
        categories = classifier.detect_categories(msg)
        assert "social" in categories

    def test_work_detected(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(sender="colleague@work.com", subject="Q3 Report")
        categories = classifier.detect_categories(msg)
        assert "work" in categories
