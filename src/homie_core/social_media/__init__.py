"""Social media profile integrations — provider abstraction, intelligence, and tools.

SocialMediaService is the main facade used by the daemon and CLI.
"""
from __future__ import annotations

import logging
from typing import Any

from homie_core.social_media.intelligence import SocialIntelligence
from homie_core.social_media.provider import (
    SocialMediaProviderBase, FeedProvider, ProfileProvider,
    PublishProvider, DirectMessageProvider,
)

logger = logging.getLogger(__name__)

# Platform config: name -> (provider_class_import_path, credential_provider_name)
_PLATFORM_CONFIG = {
    "twitter": ("homie_core.social_media.twitter_provider", "TwitterProvider", "twitter"),
    "reddit": ("homie_core.social_media.reddit_provider", "RedditProvider", "reddit"),
    "linkedin": ("homie_core.social_media.linkedin_provider", "LinkedInProvider", "linkedin"),
    "facebook": ("homie_core.social_media.facebook_provider", "FacebookProvider", "facebook"),
    "instagram": ("homie_core.social_media.instagram_provider", "InstagramProvider", "instagram"),
    "blog": ("homie_core.social_media.blog_provider", "BlogProvider", "blog"),
}


class SocialMediaService:
    """High-level facade for social media operations."""

    def __init__(self, vault, working_memory=None):
        self._vault = vault
        self._working_memory = working_memory
        self._providers: dict[str, SocialMediaProviderBase] = {}
        self._intelligence = SocialIntelligence(vault)

    def initialize(self) -> list[str]:
        """Connect all platforms with stored credentials. Returns connected platform names."""
        import importlib
        connected = []
        for platform, (module_path, class_name, cred_name) in _PLATFORM_CONFIG.items():
            try:
                creds = self._vault.list_credentials(cred_name)
                for cred in creds:
                    if not cred.active:
                        continue
                    if cred.account_id == "oauth_client":
                        continue
                    try:
                        mod = importlib.import_module(module_path)
                        provider_cls = getattr(mod, class_name)
                        provider = provider_cls()
                        if provider.connect(cred):
                            self._providers[platform] = provider
                            self._vault.set_connection_status(
                                cred_name, connected=True, label=cred.account_id,
                            )
                            connected.append(platform)
                            break
                    except Exception:
                        logger.exception("Failed to connect %s account %s", platform, cred.account_id)
            except Exception:
                pass  # No credentials for this platform
        return connected

    def sync_tick(self) -> str:
        """Called by SyncManager — fetch notifications, update working memory."""
        parts = []
        for platform, provider in self._providers.items():
            try:
                if isinstance(provider, FeedProvider):
                    posts = provider.get_feed(limit=5)
                    if posts and self._working_memory is not None:
                        summaries = [
                            {"platform": platform, "author": p.author,
                             "content": p.content[:100], "type": p.post_type}
                            for p in posts[:5]
                        ]
                        self._working_memory.update(f"sm_{platform}_feed", summaries)
                    parts.append(f"{platform}: {len(posts)} new post(s)")
            except Exception as exc:
                parts.append(f"{platform}: error ({exc})")
        return "; ".join(parts) if parts else "No social media platforms connected"

    def get_feed(self, platform: str = "all", limit: int = 20) -> list[dict]:
        results: list[dict] = []
        for plat, provider in self._providers.items():
            if platform != "all" and plat != platform:
                continue
            if isinstance(provider, FeedProvider):
                try:
                    posts = provider.get_feed(limit=limit)
                    results.extend(p.to_dict() for p in posts)
                except Exception:
                    pass
        return results[:limit]

    def get_profile(self, platform: str, username: str | None = None) -> dict:
        provider = self._providers.get(platform)
        if not provider or not isinstance(provider, ProfileProvider):
            return {"error": f"No profile provider for {platform}"}
        try:
            return provider.get_profile(username).to_dict()
        except Exception as exc:
            return {"error": str(exc)}

    def scan_profiles(self) -> dict:
        result = self._intelligence.analyze_profiles(self._providers)
        return {
            name: {"topics": p.topics, "audience_size": p.audience_size,
                   "engagement_rate": p.engagement_rate, "tone": p.tone}
            for name, p in result.platforms.items()
        }

    def publish(self, platform: str, content: str, media_urls: list[str] | None = None) -> dict:
        provider = self._providers.get(platform)
        if not provider or not isinstance(provider, PublishProvider):
            return {"status": "error", "error": f"No publish provider for {platform}"}
        try:
            return provider.publish(content, media_urls=media_urls)
        except Exception as exc:
            return {"status": "error", "platform": platform, "error": str(exc)}

    def get_conversations(self, platform: str, limit: int = 20) -> list[dict]:
        provider = self._providers.get(platform)
        if not provider or not isinstance(provider, DirectMessageProvider):
            return []
        try:
            return [c.to_dict() for c in provider.list_conversations(limit=limit)]
        except Exception:
            return []

    def get_dms(self, platform: str, conversation_id: str, limit: int = 20) -> list[dict]:
        provider = self._providers.get(platform)
        if not provider or not isinstance(provider, DirectMessageProvider):
            return []
        try:
            return [m.to_dict() for m in provider.get_messages(conversation_id, limit=limit)]
        except Exception:
            return []

    def send_dm(self, platform: str, recipient: str, text: str) -> dict:
        provider = self._providers.get(platform)
        if not provider or not isinstance(provider, DirectMessageProvider):
            return {"status": "error", "error": f"No DM provider for {platform}"}
        try:
            return provider.send_message(recipient, text)
        except Exception as exc:
            return {"status": "error", "platform": platform, "error": str(exc)}

    def get_social_profile(self) -> dict:
        cached = self._intelligence.get_cached_profile()
        if not cached:
            return {"error": "No profile scan available. Run sm_scan_profiles first."}
        return {
            "primary_topics": cached.cross_platform.primary_topics,
            "platforms": {
                name: {"topics": p.topics, "tone": p.tone,
                       "audience_size": p.audience_size}
                for name, p in cached.platforms.items()
            },
        }

    def search(self, query: str, platform: str = "all", limit: int = 10) -> list[dict]:
        results: list[dict] = []
        for plat, provider in self._providers.items():
            if platform != "all" and plat != platform:
                continue
            if isinstance(provider, FeedProvider):
                try:
                    posts = provider.search_posts(query, limit=limit)
                    results.extend(p.to_dict() for p in posts)
                except Exception:
                    pass
        return results[:limit]

    def get_notifications(self, platform: str = "all", limit: int = 20) -> list[dict]:
        # Notifications come from feed scanning — return from working memory if available
        if self._working_memory is None:
            return []
        notifications = []
        for plat in self._providers:
            if platform != "all" and plat != platform:
                continue
            data = self._working_memory.get(f"sm_{plat}_feed") if hasattr(self._working_memory, 'get') else None
            if data:
                notifications.extend(data[:limit])
        return notifications[:limit]
