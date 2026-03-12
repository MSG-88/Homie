"""Abstract social/messaging provider interface.

Slack implements this first. Future providers (Discord, Teams) implement
the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from homie_core.social.models import SocialChannel, SocialMessage


class SocialProvider(ABC):
    """Abstract social provider — one implementation per service."""

    @abstractmethod
    def connect(self, credential) -> bool:
        """Connect using a stored Credential from vault. Returns True on success."""

    @abstractmethod
    def list_channels(self) -> list[SocialChannel]:
        """List channels the authenticated user can see."""

    @abstractmethod
    def get_recent_messages(self, channel_id: str, limit: int = 20) -> list[SocialMessage]:
        """Get recent messages from a channel."""

    @abstractmethod
    def search_messages(self, query: str, limit: int = 10) -> list[SocialMessage]:
        """Search messages across the workspace."""

    @abstractmethod
    def send_message(self, channel_id: str, text: str, thread_id: str | None = None) -> str:
        """Send a message. Returns the message timestamp/ID."""

    @abstractmethod
    def get_unread_mentions(self) -> list[SocialMessage]:
        """Get unread messages that mention the authenticated user."""
