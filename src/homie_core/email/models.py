"""Email data models.

All models are plain dataclasses with no external dependencies.
Serialization uses to_dict/from_dict for database storage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmailMessage:
    """A single email message with Homie-assigned metadata."""
    id: str
    thread_id: str
    account_id: str
    provider: str
    subject: str
    sender: str
    recipients: list[str]
    snippet: str
    body: str | None = None
    labels: list[str] = field(default_factory=list)
    date: float = 0.0
    is_read: bool = True
    is_starred: bool = False
    has_attachments: bool = False
    attachment_names: list[str] = field(default_factory=list)
    priority: str = "medium"
    spam_score: float = 0.0
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "account_id": self.account_id,
            "provider": self.provider,
            "subject": self.subject,
            "sender": self.sender,
            "recipients": self.recipients,
            "snippet": self.snippet,
            "body": self.body,
            "labels": self.labels,
            "date": self.date,
            "is_read": self.is_read,
            "is_starred": self.is_starred,
            "has_attachments": self.has_attachments,
            "attachment_names": self.attachment_names,
            "priority": self.priority,
            "spam_score": self.spam_score,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmailMessage:
        return cls(
            id=data["id"],
            thread_id=data["thread_id"],
            account_id=data["account_id"],
            provider=data["provider"],
            subject=data["subject"],
            sender=data["sender"],
            recipients=data.get("recipients", []),
            snippet=data.get("snippet", ""),
            body=data.get("body"),
            labels=data.get("labels", []),
            date=data.get("date", 0.0),
            is_read=data.get("is_read", True),
            is_starred=data.get("is_starred", False),
            has_attachments=data.get("has_attachments", False),
            attachment_names=data.get("attachment_names", []),
            priority=data.get("priority", "medium"),
            spam_score=data.get("spam_score", 0.0),
            categories=data.get("categories", []),
        )


@dataclass
class EmailThread:
    """A conversation thread grouping multiple messages."""
    id: str
    account_id: str
    subject: str
    participants: list[str]
    message_count: int
    last_message_date: float
    snippet: str
    labels: list[str] = field(default_factory=list)
    messages: list[EmailMessage] = field(default_factory=list)


@dataclass
class HistoryChange:
    """A single change from Gmail's history API."""
    message_id: str
    change_type: str  # "added", "deleted", "labelAdded", "labelRemoved"
    labels: list[str] = field(default_factory=list)


@dataclass
class SyncState:
    """Tracks sync progress for one account."""
    account_id: str
    provider: str
    history_id: str | None = None
    last_full_sync: float = 0.0
    last_incremental_sync: float = 0.0
    total_synced: int = 0


@dataclass
class Label:
    """An email label/folder."""
    id: str
    name: str
    type: str = "user"  # "system" or "user"


@dataclass
class EmailSyncConfig:
    """Per-account sync and notification settings."""
    account_id: str
    check_interval: int = 300
    notify_priority: str = "high"
    quiet_hours_start: int | None = None
    quiet_hours_end: int | None = None
    auto_trash_spam: bool = True


@dataclass
class SyncResult:
    """Result of a sync operation."""
    account_id: str
    new_messages: int = 0
    updated_messages: int = 0
    trashed_messages: int = 0
    notifications: list[EmailMessage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class EmailDraft:
    """A draft email with metadata."""
    id: str
    message: EmailMessage
    updated_at: float


@dataclass
class EmailAttachment:
    """An email attachment with optional downloaded data."""
    id: str
    message_id: str
    filename: str
    mime_type: str
    size: int  # bytes
    data: bytes | None = None  # populated on download


@dataclass
class ContactInsight:
    """Aggregated contact intelligence from email patterns."""
    email: str
    name: str
    organization: str
    relationship: str  # "colleague", "client", "vendor", etc.
    email_count: int
    last_contact: float
    topics: list[str]
    pending_actions: list[str]


@dataclass
class ActionItem:
    """An action item extracted from an email."""
    id: str
    message_id: str
    thread_id: str
    description: str
    assignee: str
    deadline: float | None
    urgency: str  # "high", "medium", "low"
    status: str  # "pending", "done", "expired"
    extracted_at: float
