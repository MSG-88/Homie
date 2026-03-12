"""Tests for social media provider ABCs."""
from unittest.mock import MagicMock

from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
    PublishProvider, DirectMessageProvider,
)
from homie_core.social_media.models import ProfileInfo, ProfileStats


class ConcreteAll(
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
    PublishProvider, DirectMessageProvider,
):
    platform_name = "test"

    def get_feed(self, limit=20):
        return []

    def search_posts(self, query, limit=10):
        return []

    def get_profile(self, username=None):
        return ProfileInfo(platform="test", username="u", display_name="U", bio="")

    def get_stats(self):
        return ProfileStats(platform="test")

    def publish(self, content, media_urls=None):
        return {"status": "ok"}

    def list_conversations(self, limit=20):
        return []

    def get_messages(self, conversation_id, limit=20):
        return []

    def send_message(self, recipient, text):
        return {"status": "sent"}


class TestProviderBase:
    def test_initial_state(self):
        p = ConcreteAll()
        assert p.is_connected is False
        assert p._token is None

    def test_connect_success(self):
        p = ConcreteAll()
        cred = MagicMock()
        cred.access_token = "tok123"
        result = p.connect(cred)
        assert result is True
        assert p.is_connected is True
        assert p._token == "tok123"

    def test_connect_failure(self):
        p = ConcreteAll()
        cred = MagicMock(spec=[])  # no access_token attribute
        result = p.connect(cred)
        assert result is False

    def test_refresh_token_default(self):
        p = ConcreteAll()
        assert p.refresh_token() is False


class TestCapabilityCheck:
    def test_all_capabilities(self):
        p = ConcreteAll()
        assert isinstance(p, FeedProvider)
        assert isinstance(p, ProfileProvider)
        assert isinstance(p, PublishProvider)
        assert isinstance(p, DirectMessageProvider)


class FeedOnly(SocialMediaProviderBase, FeedProvider):
    platform_name = "feedonly"

    def get_feed(self, limit=20):
        return []

    def search_posts(self, query, limit=10):
        return []


class TestPartialCapabilities:
    def test_feed_only(self):
        p = FeedOnly()
        assert isinstance(p, FeedProvider)
        assert not isinstance(p, PublishProvider)
        assert not isinstance(p, DirectMessageProvider)
