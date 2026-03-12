"""Tests for email data models."""
from __future__ import annotations

import json

from homie_core.email.models import (
    EmailMessage,
    EmailThread,
    HistoryChange,
    Label,
    SyncState,
    EmailSyncConfig,
    SyncResult,
)


class TestEmailMessage:
    def test_create_minimal(self):
        msg = EmailMessage(
            id="msg1",
            thread_id="t1",
            account_id="user@gmail.com",
            provider="gmail",
            subject="Hello",
            sender="Alice <alice@example.com>",
            recipients=["user@gmail.com"],
            snippet="Hey there...",
        )
        assert msg.id == "msg1"
        assert msg.provider == "gmail"
        assert msg.body is None
        assert msg.priority == "medium"
        assert msg.spam_score == 0.0
        assert msg.is_read is True
        assert msg.categories == []
        assert msg.labels == []

    def test_create_full(self):
        msg = EmailMessage(
            id="msg2",
            thread_id="t2",
            account_id="user@gmail.com",
            provider="gmail",
            subject="Invoice #123",
            sender="billing@util.com",
            recipients=["user@gmail.com", "boss@work.com"],
            snippet="Your invoice is ready",
            body="Full body text here",
            labels=["INBOX", "IMPORTANT"],
            date=1710288000.0,
            is_read=False,
            is_starred=True,
            has_attachments=True,
            attachment_names=["invoice.pdf"],
            priority="high",
            spam_score=0.1,
            categories=["bill"],
        )
        assert msg.is_starred is True
        assert msg.has_attachments is True
        assert msg.attachment_names == ["invoice.pdf"]
        assert msg.priority == "high"

    def test_to_dict(self):
        msg = EmailMessage(
            id="msg1", thread_id="t1", account_id="a@b.com",
            provider="gmail", subject="Hi", sender="x@y.com",
            recipients=["a@b.com"], snippet="...",
        )
        d = msg.to_dict()
        assert d["id"] == "msg1"
        assert d["provider"] == "gmail"
        assert isinstance(d["recipients"], list)

    def test_from_dict(self):
        data = {
            "id": "msg1", "thread_id": "t1", "account_id": "a@b.com",
            "provider": "gmail", "subject": "Hi", "sender": "x@y.com",
            "recipients": ["a@b.com"], "snippet": "...",
        }
        msg = EmailMessage.from_dict(data)
        assert msg.id == "msg1"
        assert msg.recipients == ["a@b.com"]


class TestEmailThread:
    def test_create(self):
        thread = EmailThread(
            id="t1", account_id="user@gmail.com", subject="Discussion",
            participants=["alice@x.com", "bob@y.com"],
            message_count=5, last_message_date=1710288000.0,
            snippet="Latest reply...",
        )
        assert thread.message_count == 5
        assert thread.labels == []


class TestHistoryChange:
    def test_create(self):
        change = HistoryChange(
            message_id="msg1", change_type="added",
        )
        assert change.labels == []

    def test_with_labels(self):
        change = HistoryChange(
            message_id="msg1", change_type="labelAdded",
            labels=["INBOX", "IMPORTANT"],
        )
        assert len(change.labels) == 2


class TestSyncState:
    def test_defaults(self):
        state = SyncState(account_id="user@gmail.com", provider="gmail")
        assert state.history_id is None
        assert state.last_full_sync == 0.0
        assert state.total_synced == 0


class TestLabel:
    def test_defaults(self):
        label = Label(id="Label_1", name="Homie/Bills")
        assert label.type == "user"


class TestEmailSyncConfig:
    def test_defaults(self):
        config = EmailSyncConfig(account_id="user@gmail.com")
        assert config.check_interval == 300
        assert config.notify_priority == "high"
        assert config.quiet_hours_start is None
        assert config.auto_trash_spam is True

    def test_custom(self):
        config = EmailSyncConfig(
            account_id="user@gmail.com",
            check_interval=600,
            notify_priority="medium",
            quiet_hours_start=22,
            quiet_hours_end=7,
            auto_trash_spam=False,
        )
        assert config.check_interval == 600
        assert config.quiet_hours_start == 22


class TestSyncResult:
    def test_defaults(self):
        result = SyncResult(account_id="user@gmail.com")
        assert result.new_messages == 0
        assert result.notifications == []
        assert result.errors == []

    def test_with_data(self):
        msg = EmailMessage(
            id="m1", thread_id="t1", account_id="user@gmail.com",
            provider="gmail", subject="Urgent", sender="boss@work.com",
            recipients=["user@gmail.com"], snippet="Need this ASAP",
        )
        result = SyncResult(
            account_id="user@gmail.com",
            new_messages=3, notifications=[msg],
        )
        assert result.new_messages == 3
        assert len(result.notifications) == 1
