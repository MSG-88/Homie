"""Social media data models.

All models are plain dataclasses with no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SocialPost:
    """A single post on a social media platform."""

    id: str
    platform: str  # "twitter", "linkedin", "instagram", etc.
    author: str
    content: str
    timestamp: float
    url: str | None = None
    media_urls: list[str] = field(default_factory=list)
    likes: int = 0
    comments: int = 0
    shares: int = 0
    post_type: str = "text"  # "text", "image", "video", "link", etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "author": self.author,
            "content": self.content,
            "timestamp": self.timestamp,
            "url": self.url,
            "media_urls": self.media_urls,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "post_type": self.post_type,
        }


@dataclass
class ProfileInfo:
    """Basic profile information for a social media account."""

    platform: str
    username: str
    display_name: str
    bio: str
    avatar_url: str | None = None
    profile_url: str | None = None
    joined: float | None = None  # Unix timestamp
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "username": self.username,
            "display_name": self.display_name,
            "bio": self.bio,
            "avatar_url": self.avatar_url,
            "profile_url": self.profile_url,
            "joined": self.joined,
            "verified": self.verified,
        }


@dataclass
class ProfileStats:
    """Aggregated statistics for a social media profile."""

    platform: str
    followers: int = 0
    following: int = 0
    post_count: int = 0
    engagement_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "followers": self.followers,
            "following": self.following,
            "post_count": self.post_count,
            "engagement_rate": self.engagement_rate,
        }


@dataclass
class Notification:
    """A notification from a social media platform."""

    id: str
    platform: str
    type: str  # "like", "comment", "follow", "mention", "share", etc.
    sender: str
    content: str
    timestamp: float
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "type": self.type,
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp,
            "url": self.url,
        }


@dataclass
class Conversation:
    """A direct-message conversation thread on a social media platform."""

    id: str
    platform: str
    participants: list[str] = field(default_factory=list)
    last_message_preview: str = ""
    last_activity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "participants": self.participants,
            "last_message_preview": self.last_message_preview,
            "last_activity": self.last_activity,
        }


@dataclass
class DirectMessage:
    """A single direct message within a conversation."""

    id: str
    platform: str
    conversation_id: str
    sender: str
    content: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "conversation_id": self.conversation_id,
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp,
        }
