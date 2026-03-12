"""Unit tests for social_media data models."""
from __future__ import annotations

import pytest

from homie_core.social_media.models import (
    Conversation,
    DirectMessage,
    Notification,
    ProfileInfo,
    ProfileStats,
    SocialPost,
)


# ---------------------------------------------------------------------------
# SocialPost
# ---------------------------------------------------------------------------

class TestSocialPost:
    def _make(self, **kwargs) -> SocialPost:
        defaults = dict(
            id="p1",
            platform="twitter",
            author="alice",
            content="Hello world",
            timestamp=1_700_000_000.0,
        )
        defaults.update(kwargs)
        return SocialPost(**defaults)

    def test_required_fields(self):
        post = self._make()
        assert post.id == "p1"
        assert post.platform == "twitter"
        assert post.author == "alice"
        assert post.content == "Hello world"
        assert post.timestamp == 1_700_000_000.0

    def test_default_values(self):
        post = self._make()
        assert post.url is None
        assert post.media_urls == []
        assert post.likes == 0
        assert post.comments == 0
        assert post.shares == 0
        assert post.post_type == "text"

    def test_media_urls_not_shared(self):
        a = self._make()
        b = self._make()
        a.media_urls.append("http://img.example.com/1.jpg")
        assert b.media_urls == []

    def test_to_dict_keys(self):
        post = self._make(url="https://t.co/abc", likes=42, post_type="image")
        d = post.to_dict()
        assert set(d.keys()) == {
            "id", "platform", "author", "content", "timestamp",
            "url", "media_urls", "likes", "comments", "shares", "post_type",
        }

    def test_to_dict_values(self):
        post = self._make(url="https://t.co/abc", likes=10, shares=3)
        d = post.to_dict()
        assert d["url"] == "https://t.co/abc"
        assert d["likes"] == 10
        assert d["shares"] == 3
        assert d["comments"] == 0


# ---------------------------------------------------------------------------
# ProfileInfo
# ---------------------------------------------------------------------------

class TestProfileInfo:
    def _make(self, **kwargs) -> ProfileInfo:
        defaults = dict(
            platform="linkedin",
            username="bob",
            display_name="Bob Smith",
            bio="Software engineer",
        )
        defaults.update(kwargs)
        return ProfileInfo(**defaults)

    def test_required_fields(self):
        p = self._make()
        assert p.platform == "linkedin"
        assert p.username == "bob"
        assert p.display_name == "Bob Smith"
        assert p.bio == "Software engineer"

    def test_default_values(self):
        p = self._make()
        assert p.avatar_url is None
        assert p.profile_url is None
        assert p.joined is None
        assert p.verified is False

    def test_to_dict_keys(self):
        p = self._make()
        assert set(p.to_dict().keys()) == {
            "platform", "username", "display_name", "bio",
            "avatar_url", "profile_url", "joined", "verified",
        }

    def test_to_dict_values(self):
        p = self._make(
            avatar_url="https://example.com/avatar.png",
            verified=True,
            joined=1_600_000_000.0,
        )
        d = p.to_dict()
        assert d["avatar_url"] == "https://example.com/avatar.png"
        assert d["verified"] is True
        assert d["joined"] == 1_600_000_000.0


# ---------------------------------------------------------------------------
# ProfileStats
# ---------------------------------------------------------------------------

class TestProfileStats:
    def _make(self, **kwargs) -> ProfileStats:
        defaults = dict(platform="instagram")
        defaults.update(kwargs)
        return ProfileStats(**defaults)

    def test_required_fields(self):
        s = self._make()
        assert s.platform == "instagram"

    def test_default_values(self):
        s = self._make()
        assert s.followers == 0
        assert s.following == 0
        assert s.post_count == 0
        assert s.engagement_rate == 0.0

    def test_to_dict_keys(self):
        s = self._make()
        assert set(s.to_dict().keys()) == {
            "platform", "followers", "following", "post_count", "engagement_rate",
        }

    def test_to_dict_values(self):
        s = self._make(followers=1000, following=200, post_count=50, engagement_rate=3.5)
        d = s.to_dict()
        assert d["followers"] == 1000
        assert d["following"] == 200
        assert d["post_count"] == 50
        assert d["engagement_rate"] == 3.5


# ---------------------------------------------------------------------------
# Notification
# ---------------------------------------------------------------------------

class TestNotification:
    def _make(self, **kwargs) -> Notification:
        defaults = dict(
            id="n1",
            platform="twitter",
            type="like",
            sender="carol",
            content="Carol liked your post",
            timestamp=1_700_000_100.0,
        )
        defaults.update(kwargs)
        return Notification(**defaults)

    def test_required_fields(self):
        n = self._make()
        assert n.id == "n1"
        assert n.platform == "twitter"
        assert n.type == "like"
        assert n.sender == "carol"
        assert n.content == "Carol liked your post"
        assert n.timestamp == 1_700_000_100.0

    def test_default_url_is_none(self):
        n = self._make()
        assert n.url is None

    def test_to_dict_keys(self):
        n = self._make()
        assert set(n.to_dict().keys()) == {
            "id", "platform", "type", "sender", "content", "timestamp", "url",
        }

    def test_to_dict_url(self):
        n = self._make(url="https://twitter.com/post/123")
        assert n.to_dict()["url"] == "https://twitter.com/post/123"


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class TestConversation:
    def _make(self, **kwargs) -> Conversation:
        defaults = dict(
            id="c1",
            platform="instagram",
        )
        defaults.update(kwargs)
        return Conversation(**defaults)

    def test_required_fields(self):
        c = self._make()
        assert c.id == "c1"
        assert c.platform == "instagram"

    def test_default_values(self):
        c = self._make()
        assert c.participants == []
        assert c.last_message_preview == ""
        assert c.last_activity == 0.0

    def test_participants_not_shared(self):
        a = self._make()
        b = self._make()
        a.participants.append("dave")
        assert b.participants == []

    def test_to_dict_keys(self):
        c = self._make()
        assert set(c.to_dict().keys()) == {
            "id", "platform", "participants", "last_message_preview", "last_activity",
        }

    def test_to_dict_values(self):
        c = self._make(
            participants=["alice", "bob"],
            last_message_preview="Hey!",
            last_activity=1_700_000_200.0,
        )
        d = c.to_dict()
        assert d["participants"] == ["alice", "bob"]
        assert d["last_message_preview"] == "Hey!"
        assert d["last_activity"] == 1_700_000_200.0


# ---------------------------------------------------------------------------
# DirectMessage
# ---------------------------------------------------------------------------

class TestDirectMessage:
    def _make(self, **kwargs) -> DirectMessage:
        defaults = dict(
            id="dm1",
            platform="instagram",
            conversation_id="c1",
            sender="alice",
            content="Hello!",
            timestamp=1_700_000_300.0,
        )
        defaults.update(kwargs)
        return DirectMessage(**defaults)

    def test_required_fields(self):
        dm = self._make()
        assert dm.id == "dm1"
        assert dm.platform == "instagram"
        assert dm.conversation_id == "c1"
        assert dm.sender == "alice"
        assert dm.content == "Hello!"
        assert dm.timestamp == 1_700_000_300.0

    def test_to_dict_keys(self):
        dm = self._make()
        assert set(dm.to_dict().keys()) == {
            "id", "platform", "conversation_id", "sender", "content", "timestamp",
        }

    def test_to_dict_values(self):
        dm = self._make(sender="bob", content="Hi there")
        d = dm.to_dict()
        assert d["sender"] == "bob"
        assert d["content"] == "Hi there"
        assert d["conversation_id"] == "c1"
