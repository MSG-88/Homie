"""Instagram provider using the Meta Graph API."""
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

_API = "https://graph.facebook.com/v22.0"


class InstagramProvider(
    SocialMediaProviderBase,
    FeedProvider,
    ProfileProvider,
    PublishProvider,
    DirectMessageProvider,
):
    """Instagram integration via the Meta Graph API v22.0."""

    platform_name = "instagram"

    def __init__(self):
        super().__init__()
        self._is_business: bool = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, credential) -> bool:
        """Store the token and verify it against ``/me``."""
        try:
            self._token = credential.access_token
            self._is_business = getattr(credential, "is_business", False)
            resp = self._call("GET", "/me", params={"fields": "id,username"})
            resp.raise_for_status()
            self._connected = True
            return True
        except Exception:
            logger.exception("Instagram connect failed")
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _call(self, method: str, path: str, params=None, json_body=None, retries: int = 2):
        """Issue an HTTP request to the Graph API with retry on 429."""
        url = f"{_API}{path}"
        params = params or {}
        params["access_token"] = self._token

        for attempt in range(retries + 1):
            resp = requests.request(method, url, params=params, json=json_body)
            if resp.status_code == 429 and attempt < retries:
                time.sleep(1)
                continue
            return resp
        return resp  # pragma: no cover

    # ------------------------------------------------------------------
    # FeedProvider
    # ------------------------------------------------------------------

    def get_feed(self, limit: int = 20) -> list[SocialPost]:
        fields = "id,caption,timestamp,media_type,permalink,like_count,comments_count"
        resp = self._call("GET", "/me/media", params={"fields": fields, "limit": limit})
        resp.raise_for_status()
        posts: list[SocialPost] = []
        for item in resp.json().get("data", []):
            posts.append(
                SocialPost(
                    id=item["id"],
                    platform=self.platform_name,
                    author="",
                    content=item.get("caption", ""),
                    timestamp=0.0,
                    url=item.get("permalink"),
                    likes=item.get("like_count", 0),
                    comments=item.get("comments_count", 0),
                    post_type=item.get("media_type", "IMAGE").lower(),
                )
            )
        return posts

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        # Instagram Graph API does not support public post search.
        return []

    # ------------------------------------------------------------------
    # ProfileProvider
    # ------------------------------------------------------------------

    def get_profile(self, username: str | None = None) -> ProfileInfo:
        fields = "id,username,name,biography,profile_picture_url,media_count"
        resp = self._call("GET", "/me", params={"fields": fields})
        resp.raise_for_status()
        data = resp.json()
        return ProfileInfo(
            platform=self.platform_name,
            username=data.get("username", ""),
            display_name=data.get("name", ""),
            bio=data.get("biography", ""),
            avatar_url=data.get("profile_picture_url"),
        )

    def get_stats(self) -> ProfileStats:
        resp = self._call(
            "GET", "/me", params={"fields": "followers_count,follows_count,media_count"}
        )
        resp.raise_for_status()
        data = resp.json()
        return ProfileStats(
            platform=self.platform_name,
            followers=data.get("followers_count", 0),
            following=data.get("follows_count", 0),
            post_count=data.get("media_count", 0),
        )

    # ------------------------------------------------------------------
    # PublishProvider
    # ------------------------------------------------------------------

    def publish(self, content: str, media_urls: list[str] | None = None) -> dict:
        if not self._is_business:
            return {
                "status": "error",
                "error": "Instagram publish requires a business or creator account",
            }
        if not media_urls:
            return {
                "status": "error",
                "error": "Instagram requires at least one media URL to publish",
            }
        # Step 1: create media container
        container_resp = self._call(
            "POST",
            "/me/media",
            json_body={"image_url": media_urls[0], "caption": content},
        )
        container_resp.raise_for_status()
        creation_id = container_resp.json().get("id")
        # Step 2: publish container
        publish_resp = self._call(
            "POST",
            "/me/media_publish",
            json_body={"creation_id": creation_id},
        )
        publish_resp.raise_for_status()
        return {"status": "ok", "id": publish_resp.json().get("id")}

    # ------------------------------------------------------------------
    # DirectMessageProvider
    # ------------------------------------------------------------------

    def list_conversations(self, limit: int = 20) -> list[Conversation]:
        if not self._is_business:
            return []

        resp = self._call(
            "GET",
            "/me/conversations",
            params={"platform": "instagram", "limit": limit},
        )
        resp.raise_for_status()
        convos: list[Conversation] = []
        for item in resp.json().get("data", []):
            convos.append(
                Conversation(
                    id=item["id"],
                    platform=self.platform_name,
                    participants=[
                        p.get("username", "")
                        for p in item.get("participants", {}).get("data", [])
                    ],
                    last_message_preview=item.get("snippet", ""),
                )
            )
        return convos

    def get_messages(self, conversation_id: str, limit: int = 20) -> list[DirectMessage]:
        if not self._is_business:
            return []

        resp = self._call(
            "GET",
            f"/{conversation_id}/messages",
            params={"fields": "from,message,created_time", "limit": limit},
        )
        resp.raise_for_status()
        msgs: list[DirectMessage] = []
        for item in resp.json().get("data", []):
            msgs.append(
                DirectMessage(
                    id=item.get("id", ""),
                    platform=self.platform_name,
                    conversation_id=conversation_id,
                    sender=item.get("from", {}).get("name", ""),
                    content=item.get("message", ""),
                    timestamp=0.0,
                )
            )
        return msgs

    def send_message(self, recipient: str, text: str) -> dict:
        if not self._is_business:
            return {
                "status": "error",
                "error": "Instagram DM requires a business or creator account",
            }
        resp = self._call(
            "POST", f"/{recipient}/messages", json_body={"message": text}
        )
        resp.raise_for_status()
        return {"status": "sent", "id": resp.json().get("id")}
