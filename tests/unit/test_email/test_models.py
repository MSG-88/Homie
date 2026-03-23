"""Tests for new email data models."""
from __future__ import annotations

from homie_core.email.models import (
    EmailAttachment,
    EmailDraft,
    EmailMessage,
    EmailThread,
    ActionItem,
    ContactInsight,
)


class TestEmailThreadExtension:
    def test_thread_has_messages_field(self):
        thread = EmailThread(
            id="t1", account_id="user@gmail.com", subject="Test",
            participants=["alice@x.com"], message_count=2,
            last_message_date=1710288000.0, snippet="Hello",
        )
        assert thread.messages == []

    def test_thread_with_messages(self):
        msg = EmailMessage(
            id="msg1", thread_id="t1", account_id="user@gmail.com",
            provider="gmail", subject="Test", sender="alice@x.com",
            recipients=["user@gmail.com"], snippet="Hello",
        )
        thread = EmailThread(
            id="t1", account_id="user@gmail.com", subject="Test",
            participants=["alice@x.com"], message_count=1,
            last_message_date=1710288000.0, snippet="Hello",
            messages=[msg],
        )
        assert len(thread.messages) == 1
        assert thread.messages[0].id == "msg1"


class TestEmailDraft:
    def test_draft_creation(self):
        msg = EmailMessage(
            id="msg1", thread_id="t1", account_id="user@gmail.com",
            provider="gmail", subject="Draft", sender="user@gmail.com",
            recipients=["bob@x.com"], snippet="",
        )
        draft = EmailDraft(id="d1", message=msg, updated_at=1710288000.0)
        assert draft.id == "d1"
        assert draft.message.subject == "Draft"


class TestEmailAttachment:
    def test_attachment_metadata(self):
        att = EmailAttachment(
            id="att1", message_id="msg1", filename="report.pdf",
            mime_type="application/pdf", size=1024,
        )
        assert att.data is None
        assert att.size == 1024

    def test_attachment_with_data(self):
        att = EmailAttachment(
            id="att1", message_id="msg1", filename="report.pdf",
            mime_type="application/pdf", size=5, data=b"hello",
        )
        assert att.data == b"hello"


class TestActionItem:
    def test_action_item_defaults(self):
        item = ActionItem(
            id="a1", message_id="msg1", thread_id="t1",
            description="Review PR", assignee="user@gmail.com",
            deadline=None, urgency="medium", status="pending",
            extracted_at=1710288000.0,
        )
        assert item.status == "pending"
        assert item.deadline is None


class TestContactInsight:
    def test_contact_insight(self):
        ci = ContactInsight(
            email="alice@x.com", name="Alice", organization="X Corp",
            relationship="colleague", email_count=42,
            last_contact=1710288000.0, topics=["Project Alpha"],
            pending_actions=["Review PR #123"],
        )
        assert ci.email_count == 42
        assert len(ci.topics) == 1
