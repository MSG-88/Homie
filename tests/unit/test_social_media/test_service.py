"""Tests for SocialMediaService facade."""
from unittest.mock import MagicMock, patch
from homie_core.social_media import SocialMediaService
from homie_core.social_media.models import SocialPost, ProfileInfo, ProfileStats
from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
    PublishProvider, DirectMessageProvider,
)


def _make_post(**overrides):
    defaults = dict(id="p1", platform="twitter", author="@test",
                    content="Hello", timestamp=1700000000.0)
    defaults.update(overrides)
    return SocialPost(**defaults)


class _MockFullProvider(SocialMediaProviderBase, FeedProvider, ProfileProvider,
                        PublishProvider, DirectMessageProvider):
    """Concrete mock provider that satisfies all isinstance checks."""

    def get_feed(self, limit=20):
        return []

    def search_posts(self, query, limit=10):
        return []

    def get_profile(self, username=None):
        return None

    def get_stats(self):
        return None

    def publish(self, content, media_urls=None):
        return {}

    def list_conversations(self, limit=20):
        return []

    def get_messages(self, conversation_id, limit=20):
        return []

    def send_message(self, recipient, text):
        return {}


def _make_provider(**method_overrides):
    """Create a _MockFullProvider with overridden methods via MagicMock."""
    provider = _MockFullProvider()
    for name, return_value in method_overrides.items():
        setattr(provider, name, MagicMock(return_value=return_value))
    return provider


class TestInitialize:
    def test_initialize_connects_platform(self):
        vault = MagicMock()
        cred = MagicMock()
        cred.active = True
        cred.account_id = "myaccount"
        vault.list_credentials.return_value = [cred]

        mock_provider = MagicMock()
        mock_provider.connect.return_value = True
        mock_mod = MagicMock()
        mock_mod.TwitterProvider.return_value = mock_provider
        mock_mod.RedditProvider.return_value = mock_provider
        mock_mod.LinkedInProvider.return_value = mock_provider
        mock_mod.FacebookProvider.return_value = mock_provider
        mock_mod.InstagramProvider.return_value = mock_provider
        mock_mod.BlogProvider.return_value = mock_provider

        with patch("importlib.import_module", return_value=mock_mod):
            service = SocialMediaService(vault=vault)
            connected = service.initialize()
            assert len(connected) > 0

    def test_initialize_no_credentials(self):
        vault = MagicMock()
        vault.list_credentials.return_value = []
        service = SocialMediaService(vault=vault)
        connected = service.initialize()
        assert connected == []


class TestSyncTick:
    def test_no_providers(self):
        vault = MagicMock()
        service = SocialMediaService(vault=vault)
        assert service.sync_tick() == "No social media platforms connected"

    def test_with_provider(self):
        vault = MagicMock()
        wm = MagicMock()
        provider = _make_provider(get_feed=[_make_post()])
        service = SocialMediaService(vault=vault, working_memory=wm)
        service._providers["twitter"] = provider
        result = service.sync_tick()
        assert "twitter: 1 new post(s)" in result


class TestGetFeed:
    def test_get_feed(self):
        vault = MagicMock()
        provider = _make_provider(get_feed=[_make_post()])
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        feed = service.get_feed(platform="twitter")
        assert len(feed) == 1
        assert feed[0]["content"] == "Hello"

    def test_get_feed_filters_platform(self):
        vault = MagicMock()
        provider = _make_provider(get_feed=[_make_post()])
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        feed = service.get_feed(platform="reddit")
        assert feed == []


class TestPublish:
    def test_publish_success(self):
        vault = MagicMock()
        provider = _make_provider(publish={"status": "published", "post_id": "123"})
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        result = service.publish("twitter", "Hello!")
        assert result["status"] == "published"

    def test_publish_no_provider(self):
        vault = MagicMock()
        service = SocialMediaService(vault=vault)
        result = service.publish("twitter", "Hello!")
        assert result["status"] == "error"


class TestDMs:
    def test_send_dm(self):
        vault = MagicMock()
        provider = _make_provider(send_message={"status": "sent", "message_id": "dm1"})
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        result = service.send_dm("twitter", "@user", "Hi!")
        assert result["status"] == "sent"

    def test_send_dm_no_provider(self):
        vault = MagicMock()
        service = SocialMediaService(vault=vault)
        result = service.send_dm("twitter", "@user", "Hi!")
        assert result["status"] == "error"


class TestScanProfiles:
    def test_scan(self):
        vault = MagicMock()
        vault.get_credential.return_value = None
        provider = _make_provider(
            get_stats=ProfileStats(platform="twitter", followers=100),
            get_feed=[_make_post(content="#python code")],
        )
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        result = service.scan_profiles()
        assert "twitter" in result


class TestSearch:
    def test_search(self):
        vault = MagicMock()
        provider = _make_provider(search_posts=[_make_post()])
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        results = service.search("hello")
        assert len(results) == 1

    def test_search_filters_platform(self):
        vault = MagicMock()
        provider = _make_provider()
        service = SocialMediaService(vault=vault)
        service._providers["twitter"] = provider
        results = service.search("hello", platform="reddit")
        assert results == []


class TestGetSocialProfile:
    def test_no_cache(self):
        vault = MagicMock()
        vault.get_credential.return_value = None
        service = SocialMediaService(vault=vault)
        result = service.get_social_profile()
        assert "error" in result
