"""Social/messaging data models.

All models are plain dataclasses with no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SocialMessage:
    """A single message from a social/messaging platform."""
    id: str
    platform: str  # "slack", "discord", etc.
    channel: str
    sender: str
    content: str
    timestamp: float
    thread_id: str | None = None
    is_mention: bool = False
    is_dm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "channel": self.channel,
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
            "is_mention": self.is_mention,
            "is_dm": self.is_dm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SocialMessage:
        return cls(
            id=data["id"],
            platform=data["platform"],
            channel=data["channel"],
            sender=data["sender"],
            content=data["content"],
            timestamp=data["timestamp"],
            thread_id=data.get("thread_id"),
            is_mention=data.get("is_mention", False),
            is_dm=data.get("is_dm", False),
        )


@dataclass
class SocialChannel:
    """A channel or conversation on a social platform."""
    id: str
    name: str
    platform: str
    is_dm: bool = False
    member_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "platform": self.platform,
            "is_dm": self.is_dm,
            "member_count": self.member_count,
        }


@dataclass
class SocialNotification:
    """A notification triggered by a social message."""
    message: SocialMessage
    reason: str  # "mention", "dm", "keyword"

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message.to_dict(),
            "reason": self.reason,
        }
