"""Tests for new EmailService facade methods."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

from homie_core.email import EmailService
from homie_core.email.models import EmailDraft, EmailMessage, EmailThread


def _make_service():
    vault = MagicMock()
    vault.list_credentials.return_value = []
    conn = sqlite3.connect(":memory:")
    svc = EmailService(vault=vault, cache_conn=conn)
    return svc


def _make_service_with_provider():
    svc = _make_service()
    provider = MagicMock()
    svc._providers["user@gmail.com"] = provider
    return svc, provider


class TestEmailServiceThreads:
    def test_get_thread_messages(self):
        svc, provider = _make_service_with_provider()
        thread = EmailThread(
            id="t1", account_id="user@gmail.com", subject="Test",
            participants=[], message_count=1, last_message_date=0.0, snippet="",
        )
        provider.get_thread.return_value = thread
        result = svc.get_thread_messages("t1")
        assert result.id == "t1"

    def test_list_inbox_threads(self):
        svc, provider = _make_service_with_provider()
        provider.get_inbox_threads.return_value = []
        result = svc.list_inbox_threads()
        assert result == []

    def test_get_unread_counts(self):
        svc, provider = _make_service_with_provider()
        provider.get_unread_count.side_effect = lambda label: {"INBOX": 5, "SPAM": 2, "STARRED": 1}.get(label, 0)
        counts = svc.get_unread_counts()
        assert counts["inbox"] == 5
        assert counts["spam"] == 2


class TestEmailServiceDrafts:
    def test_list_drafts(self):
        svc, provider = _make_service_with_provider()
        provider.list_drafts.return_value = []
        assert svc.list_drafts() == []

    def test_delete_draft(self):
        svc, provider = _make_service_with_provider()
        svc.delete_draft("d1")
        provider.delete_draft.assert_called_with("d1")


class TestEmailServiceSend:
    def test_send_email(self):
        svc, provider = _make_service_with_provider()
        provider.send_email.return_value = "sent1"
        result = svc.send_email(to="bob@x.com", subject="Hi", body="Hello")
        assert result == "sent1"

    def test_reply_default_draft(self):
        svc, provider = _make_service_with_provider()
        provider.reply.return_value = "draft1"
        result = svc.reply("msg1", body="Thanks")
        provider.reply.assert_called_with("msg1", "Thanks", False)

    def test_forward(self):
        svc, provider = _make_service_with_provider()
        provider.forward.return_value = "fwd1"
        result = svc.forward("msg1", to="charlie@x.com", body="FYI")
        assert result == "fwd1"


class TestEmailServiceAttachments:
    def test_get_attachments(self):
        svc, provider = _make_service_with_provider()
        provider.get_attachments.return_value = []
        result = svc.get_attachments("msg1")
        assert result == []
