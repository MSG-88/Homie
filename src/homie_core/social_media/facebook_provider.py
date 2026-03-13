"""Facebook provider using the Meta Graph API."""
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


class FacebookProvider(
    SocialMediaProviderBase,
    FeedProvider,
    ProfileProvider,
    PublishProvider,
    DirectMessageProvider,
):
    """Facebook integration via the Meta Graph API v22.0."""

    platform_name = "facebook"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, credential) -> bool:
        """Store the token and verify it against ``/me``."""
        try:
            self._token = credential.access_token
            resp = self._call("GET", "/me", params={"fields": "id,name"})
            resp.raise_for_status()
            self._connected = True
            return True
        except Exception:
            logger.exception("Facebook connect failed")
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
        fields = "id,message,created_time,from,likes.summary(true),comments.summary(true)"
        resp = self._call("GET", "/me/feed", params={"fields": fields, "limit": limit})
        resp.raise_for_status()
        posts: list[SocialPost] = []
        for item in resp.json().get("data", []):
            posts.append(
                SocialPost(
                    id=item["id"],
                    platform=self.platform_name,
                    author=item.get("from", {}).get("name", ""),
                    content=item.get("message", ""),
                    timestamp=0.0,
                    likes=item.get("likes", {}).get("summary", {}).get("total_count", 0),
                    comments=item.get("comments", {}).get("summary", {}).get("total_count", 0),
                )
            )
        return posts

    def search_posts(self, query: str, limit: int = 10) -> list[SocialPost]:
        resp = self._call(
            "GET", "/search", params={"q": query, "type": "post", "limit": limit}
        )
        resp.raise_for_status()
        posts: list[SocialPost] = []
        for item in resp.json().get("data", []):
            posts.append(
                SocialPost(
                    id=item.get("id", ""),
                    platform=self.platform_name,
                    author=item.get("from", {}).get("name", ""),
                    content=item.get("message", ""),
                    timestamp=0.0,
                )
            )
        return posts

    # ------------------------------------------------------------------
    # ProfileProvider
    # ------------------------------------------------------------------

    def get_profile(self, username: str | None = None) -> ProfileInfo:
        fields = "id,name,email,picture"
        resp = self._call("GET", "/me", params={"fields": fields})
        resp.raise_for_status()
        data = resp.json()
        return ProfileInfo(
            platform=self.platform_name,
            username=data.get("id", ""),
            display_name=data.get("name", ""),
            bio="",
            avatar_url=data.get("picture", {}).get("data", {}).get("url"),
        )

    def get_stats(self) -> ProfileStats:
        resp = self._call("GET", "/me", params={"fields": "friends.summary(true)"})
        resp.raise_for_status()
        data = resp.json()
        friend_count = (
            data.get("friends", {}).get("summary", {}).get("total_count", 0)
        )
        return ProfileStats(
            platform=self.platform_name,
            followers=friend_count,
        )

    # ------------------------------------------------------------------
    # PublishProvider
    # ------------------------------------------------------------------

    def publish(self, content: str, media_urls: list[str] | None = None) -> dict:
        resp = self._call("POST", "/me/feed", json_body={"message": content})
        resp.raise_for_status()
        return {"status": "ok", "id": resp.json().get("id")}

    # ------------------------------------------------------------------
    # DirectMessageProvider
    # ------------------------------------------------------------------

    def list_conversations(self, limit: int = 20) -> list[Conversation]:
        fields = "participants,snippet,updated_time"
        resp = self._call(
            "GET", "/me/conversations", params={"fields": fields, "limit": limit}
        )
        resp.raise_for_status()
        convos: list[Conversation] = []
        for item in resp.json().get("data", []):
            participants = [
                p.get("name", "")
                for p in item.get("participants", {}).get("data", [])
            ]
            convos.append(
                Conversation(
                    id=item["id"],
                    platform=self.platform_name,
                    participants=participants,
                    last_message_preview=item.get("snippet", ""),
                )
            )
        return convos

    def get_messages(self, conversation_id: str, limit: int = 20) -> list[DirectMessage]:
        fields = "from,message,created_time"
        resp = self._call(
            "GET",
            f"/{conversation_id}/messages",
            params={"fields": fields, "limit": limit},
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
        resp = self._call(
            "POST", f"/{recipient}/messages", json_body={"message": text}
        )
        resp.raise_for_status()
        return {"status": "sent", "id": resp.json().get("id")}
