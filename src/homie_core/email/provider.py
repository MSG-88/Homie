"""Abstract email provider interface.

Gmail implements this directly via google-api-python-client.
Future providers (Outlook, IMAP) implement the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from homie_core.email.models import (
    EmailMessage,
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
