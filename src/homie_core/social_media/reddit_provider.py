"""Reddit provider for Homie social media module.

Implements Feed, Profile, Publish, and DM capabilities via the Reddit
JSON API (https://oauth.reddit.com).
"""
from __future__ import annotations

import logging
import time

import requests

from homie_core.social_media.models import (
    Conversation,
    DirectMessage,
    ProfileInfo,
    ProfileStats,
    SocialPost,
)
from homie_core.social_media.provider import (
    DirectMessageProvider,
    FeedProvider,
    ProfileProvider,
    PublishProvider,
    SocialMediaProviderBase,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://oauth.reddit.com"


class RedditProvider(
    SocialMediaProviderBase,
    FeedProvider,
    ProfileProvider,
    PublishProvider,
    DirectMessageProvider,
):
    """Reddit social-media provider backed by the OAuth JSON API."""

    platform_name = "reddit"

    def __init__(self) -> None:
        super().__init__()
        self._username: str | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, credential) -> bool:
        """Store token, verify via ``/api/v1/me``, and cache username."""
        if not super().connect(credential):
            return False
        try:
            me = self._call("GET", "/api/v1/me")
            self._username = me["name"]
            return True
        except Exception:
            logger.exception("Reddit connect verification failed")
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
        retries: int = 2,
    ) -> dict:
        """Make an authenticated request to the Reddit API.

        Handles 429 (rate-limit) responses by backing off and retrying.
        """
        url = f"{BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": "Homie/1.0",
        }

        for attempt in range(retries + 1):
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
            )
            if resp.status_code == 429:
                if attempt < retries:
                    wait = float(resp.headers.get("Retry-After", 1))
                    time.sleep(wait)
                    continue
            resp.raise_for_status()
            return resp.json()

        # Should not reach here, but just in case:
        resp.raise_for_status()  # type: ignore[possibly-undefined]
        return resp.json()  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # FeedProvider
    # ------------------------------------------------------------------

    def get_feed(self, limit: int = 20) -> list[SocialPost]:
        data = self._call("GET", "/hot", params={"limit": limit})
        return [self._post_from_listing(child["data"]) for child in data["data"]["children"]]

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        data = self._call("GET", "/search", params={"q": query, "type": "link", "limit": limit})
        return [self._post_from_listing(child["data"]) for child in data["data"]["children"]]

    # ------------------------------------------------------------------
    # ProfileProvider
    # ------------------------------------------------------------------

    def get_profile(self, username: str | None = None) -> ProfileInfo:
        if username is None:
            data = self._call("GET", "/api/v1/me")
        else:
            data = self._call("GET", f"/user/{username}/about")
            data = data.get("data", data)

        subreddit = data.get("subreddit", {})
        return ProfileInfo(
            platform=self.platform_name,
            username=data.get("name", ""),
            display_name=subreddit.get("title", data.get("name", "")),
            bio=subreddit.get("public_description", ""),
            avatar_url=data.get("icon_img"),
            profile_url=f"https://www.reddit.com/user/{data.get('name', '')}",
            joined=data.get("created_utc"),
            verified=data.get("verified", False),
        )

    def get_stats(self) -> ProfileStats:
        data = self._call("GET", "/api/v1/me")
        return ProfileStats(
            platform=self.platform_name,
            followers=data.get("total_karma", 0),
            following=0,
            post_count=0,
            engagement_rate=0.0,
        )

    # ------------------------------------------------------------------
    # PublishProvider
    # ------------------------------------------------------------------

    def publish(self, content: str, media_urls: list[str] | None = None) -> dict:
        payload = {
            "kind": "self",
            "sr": f"u_{self._username}",
            "title": content[:300],
            "text": content,
        }
        result = self._call("POST", "/api/submit", json_body=payload)
        return result

    # ------------------------------------------------------------------
    # DirectMessageProvider
    # ------------------------------------------------------------------

    def list_conversations(self, limit: int = 20) -> list[Conversation]:
        data = self._call("GET", "/message/inbox", params={"limit": limit})
        conversations: list[Conversation] = []
        for child in data.get("data", {}).get("children", []):
            msg = child["data"]
            conversations.append(
                Conversation(
                    id=msg["name"],
                    platform=self.platform_name,
                    participants=[msg.get("author", ""), msg.get("dest", "")],
                    last_message_preview=msg.get("body", "")[:120],
                    last_activity=msg.get("created_utc", 0.0),
                )
            )
        return conversations

    def get_messages(self, conversation_id: str, limit: int = 20) -> list[DirectMessage]:
        data = self._call(
            "GET", f"/message/messages/{conversation_id}", params={"limit": limit}
        )
        messages: list[DirectMessage] = []
        for child in data.get("data", {}).get("children", []):
            msg = child["data"]
            messages.append(
                DirectMessage(
                    id=msg["name"],
                    platform=self.platform_name,
                    conversation_id=conversation_id,
                    sender=msg.get("author", ""),
                    content=msg.get("body", ""),
                    timestamp=msg.get("created_utc", 0.0),
                )
            )
        return messages

    def send_message(self, recipient: str, text: str) -> dict:
        payload = {
            "to": recipient,
            "subject": "Message from Homie",
            "text": text,
        }
        return self._call("POST", "/api/compose", json_body=payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _post_from_listing(data: dict) -> SocialPost:
        """Convert a Reddit listing child ``data`` dict to a ``SocialPost``."""
        content = data.get("selftext") or data.get("title", "")
        return SocialPost(
            id=data.get("name", ""),
            platform="reddit",
            author=data.get("author", ""),
            content=content,
            timestamp=data.get("created_utc", 0.0),
            url=f"https://www.reddit.com{data.get('permalink', '')}",
            likes=data.get("ups", 0),
            comments=data.get("num_comments", 0),
            shares=0,
            post_type="link" if data.get("is_self") is False else "text",
        )
