"""Abstract email provider interface.

Gmail implements this directly via google-api-python-client.
Future providers (Outlook, IMAP) implement the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from homie_core.email.models import (
    EmailAttachment,
    EmailDraft,
    EmailMessage,
    EmailThread,
    HistoryChange,
    Label,
)


class EmailProvider(ABC):
    """Abstract email provider — one implementation per service."""

    @abstractmethod
    def authenticate(self, credential) -> None:
        """Authenticate with stored Credential (from vault). Refreshes token if expired.

        Args:
            credential: A vault Credential dataclass instance with attributes:
                access_token, refresh_token, expires_at, scopes, account_id, etc.
        """

    @abstractmethod
    def fetch_messages(self, since: float, max_results: int = 100) -> list[EmailMessage]:
        """Fetch messages newer than `since` timestamp."""

    @abstractmethod
    def fetch_message_body(self, message_id: str) -> str:
        """Fetch full body text of a specific message."""

    @abstractmethod
    def get_history(self, start_history_id: str) -> tuple[list[HistoryChange], str]:
        """Get changes since history_id. Returns (changes, new_history_id)."""

    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> list[EmailMessage]:
        """Search messages using provider-native query syntax."""

    @abstractmethod
    def apply_label(self, message_id: str, label_id: str) -> None:
        """Apply a label to a message."""

    @abstractmethod
    def remove_label(self, message_id: str, label_id: str) -> None:
        """Remove a label from a message."""

    @abstractmethod
    def trash(self, message_id: str) -> None:
        """Move a message to trash."""

    @abstractmethod
    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        reply_to: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> str:
        """Create a draft email. Returns draft ID. Never sends."""

    @abstractmethod
    def list_labels(self) -> list[Label]:
        """List all labels/folders for the account."""

    @abstractmethod
    def get_profile(self) -> dict:
        """Get account profile (email address, display name)."""

    @abstractmethod
    def mark_read(self, message_id: str) -> None:
        """Mark a message as read."""

    @abstractmethod
    def archive(self, message_id: str) -> None:
        """Archive a message (remove from inbox)."""

    @abstractmethod
    def fetch_message(self, message_id: str) -> EmailMessage:
        """Fetch and parse a single message by ID."""

    @abstractmethod
    def create_label(self, name: str, visibility: str = "labelShow") -> Label:
        """Create a new label. Returns the created Label."""

    # ── Thread operations ────────────────────────────────────────────

    @abstractmethod
    def get_thread(self, thread_id: str) -> EmailThread:
        """Fetch full thread with all messages."""

    @abstractmethod
    def list_threads(self, query: str, max_results: int = 20) -> list[EmailThread]:
        """Search/list threads."""

    @abstractmethod
    def get_inbox_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        """Get threads in INBOX."""

    @abstractmethod
    def get_starred_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        """Get starred threads."""

    @abstractmethod
    def get_spam_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        """Get spam threads."""

    @abstractmethod
    def get_trash_threads(self, start: int = 0, max_results: int = 20) -> list[EmailThread]:
        """Get trashed threads."""

    @abstractmethod
    def get_unread_count(self, label: str = "INBOX") -> int:
        """Get unread message count for a label."""

    @abstractmethod
    def archive_thread(self, thread_id: str) -> None:
        """Archive entire thread (remove INBOX label)."""

    @abstractmethod
    def trash_thread(self, thread_id: str) -> None:
        """Move entire thread to trash."""

    @abstractmethod
    def apply_label_to_thread(self, thread_id: str, label_id: str) -> None:
        """Apply label to all messages in thread."""

    @abstractmethod
    def mark_thread_read(self, thread_id: str) -> None:
        """Mark entire thread as read."""

    @abstractmethod
    def mark_thread_unread(self, thread_id: str) -> None:
        """Mark entire thread as unread."""

    # ── Draft management ─────────────────────────────────────────────

    @abstractmethod
    def list_drafts(self, max_results: int = 20) -> list[EmailDraft]:
        """List all drafts."""

    @abstractmethod
    def get_draft(self, draft_id: str) -> EmailDraft:
        """Get a single draft with message content."""

    @abstractmethod
    def update_draft(self, draft_id: str, to: str, subject: str, body: str,
                     cc: list[str] | None = None, bcc: list[str] | None = None) -> str:
        """Update an existing draft. Returns draft ID."""

    @abstractmethod
    def delete_draft(self, draft_id: str) -> None:
        """Permanently delete a draft."""

    # ── Send / Reply / Forward ───────────────────────────────────────

    @abstractmethod
    def send_email(self, to: str, subject: str, body: str,
                   cc: list[str] | None = None, bcc: list[str] | None = None,
                   attachments: list[str] | None = None,
                   reply_to_message_id: str | None = None) -> str:
        """Send email directly. Returns message ID."""

    @abstractmethod
    def send_draft(self, draft_id: str) -> str:
        """Send an existing draft. Returns message ID."""

    @abstractmethod
    def reply(self, message_id: str, body: str, send: bool = False) -> str:
        """Reply to a message. Draft by default. Returns draft/message ID."""

    @abstractmethod
    def reply_all(self, message_id: str, body: str, send: bool = False) -> str:
        """Reply-all. Draft by default. Returns draft/message ID."""

    @abstractmethod
    def forward(self, message_id: str, to: str, body: str, send: bool = False) -> str:
        """Forward a message. Draft by default. Returns draft/message ID."""

    # ── Attachments ──────────────────────────────────────────────────

    @abstractmethod
    def get_attachments(self, message_id: str) -> list[EmailAttachment]:
        """List attachment metadata for a message."""

    @abstractmethod
    def download_attachment(self, message_id: str, attachment_id: str,
                            save_path: str) -> str:
        """Download attachment to local storage. Returns file path."""

    # ── Misc ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_aliases(self) -> list[str]:
        """List send-as aliases."""

    @abstractmethod
    def star(self, message_id: str) -> None:
        """Star a message."""

    @abstractmethod
    def unstar(self, message_id: str) -> None:
        """Unstar a message."""

    @abstractmethod
    def mark_unread(self, message_id: str) -> None:
        """Mark a message as unread."""

    @abstractmethod
    def move_to_inbox(self, message_id: str) -> None:
        """Move message back to inbox (undo archive)."""

    @abstractmethod
    def delete_label(self, label_id: str) -> None:
        """Delete a user label."""

    @abstractmethod
    def update_label(self, label_id: str, new_name: str) -> Label:
        """Rename a label. Returns updated Label."""

    @abstractmethod
    def untrash(self, message_id: str) -> None:
        """Restore a message from trash."""
