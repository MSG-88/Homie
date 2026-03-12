"""Abstract base classes for social media providers."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from homie_core.social_media.models import (
    SocialPost, ProfileInfo, ProfileStats, Conversation, DirectMessage,
)

logger = logging.getLogger(__name__)


class SocialMediaProviderBase:
    """Base class all providers inherit — handles credential lifecycle."""

    platform_name: str = ""

    def __init__(self):
        self._token: str | None = None
        self._connected: bool = False

    def connect(self, credential) -> bool:
        try:
            self._token = credential.access_token
            self._connected = True
            return True
        except Exception:
            logger.exception("Failed to connect %s", self.platform_name)
            return False

    def refresh_token(self) -> bool:
        return False

    @property
    def is_connected(self) -> bool:
        return self._connected


class FeedProvider(ABC):
    @abstractmethod
    def get_feed(self, limit: int = 20) -> list[SocialPost]: ...

    @abstractmethod
    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]: ...


class ProfileProvider(ABC):
    @abstractmethod
    def get_profile(self, username: str | None = None) -> ProfileInfo: ...

    @abstractmethod
    def get_stats(self) -> ProfileStats: ...


class PublishProvider(ABC):
    @abstractmethod
    def publish(self, content: str, media_urls: list[str] | None = None) -> dict: ...


class DirectMessageProvider(ABC):
    @abstractmethod
    def list_conversations(self, limit: int = 20) -> list[Conversation]: ...

    @abstractmethod
    def get_messages(self, conversation_id: str, limit: int = 20) -> list[DirectMessage]: ...

    @abstractmethod
    def send_message(self, recipient: str, text: str) -> dict: ...
