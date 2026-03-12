"""Slack implementation of SocialProvider.

Uses Slack Web API via requests library with Bearer token auth.
"""
from __future__ import annotations

import logging
import time

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from homie_core.social.models import SocialChannel, SocialMessage
from homie_core.social.provider import SocialProvider

logger = logging.getLogger(__name__)


class SlackProvider(SocialProvider):
    """Slack Web API provider."""

    BASE_URL = "https://slack.com/api"

    def __init__(self, account_id: str):
        self._account_id = account_id
        self._token: str | None = None
        self._team_name: str | None = None
        self._user_id: str | None = None

    def connect(self, credential) -> bool:
        """Validate token via auth.test and store connection info."""
        self._token = credential.access_token
        info = self._call("auth.test")
        if not info.get("ok", False):
            return False
        self._team_name = info.get("team")
        self._user_id = info.get("user_id")
        return True

    def list_channels(self) -> list[SocialChannel]:
        """List channels via conversations.list."""
        result = self._call(
            "conversations.list",
            types="public_channel,private_channel,mpim,im",
            limit=200,
        )
        channels = []
        for ch in result.get("channels", []):
            channels.append(SocialChannel(
                id=ch["id"],
                name=ch.get("name", ch.get("user", "dm")),
                platform="slack",
                is_dm=ch.get("is_im", False),
                member_count=ch.get("num_members", 0),
            ))
        return channels

    def get_recent_messages(self, channel_id: str, limit: int = 20) -> list[SocialMessage]:
        """Get recent messages via conversations.history."""
        result = self._call("conversations.history", channel=channel_id, limit=limit)
        messages = []
        for msg in result.get("messages", []):
            messages.append(SocialMessage(
                id=msg.get("ts", ""),
                platform="slack",
                channel=channel_id,
                sender=msg.get("user", "unknown"),
                content=msg.get("text", ""),
                timestamp=float(msg.get("ts", 0)),
                thread_id=msg.get("thread_ts"),
                is_mention=self._user_id is not None and f"<@{self._user_id}>" in msg.get("text", ""),
                is_dm=False,
            ))
        return messages

    def search_messages(self, query: str, limit: int = 10) -> list[SocialMessage]:
        """Search messages via search.messages."""
        result = self._call("search.messages", query=query, count=limit)
        messages = []
        matches = result.get("messages", {}).get("matches", [])
        for msg in matches:
            channel_info = msg.get("channel", {})
            messages.append(SocialMessage(
                id=msg.get("ts", ""),
                platform="slack",
                channel=channel_info.get("id", ""),
                sender=msg.get("user", msg.get("username", "unknown")),
                content=msg.get("text", ""),
                timestamp=float(msg.get("ts", 0)),
                thread_id=msg.get("thread_ts"),
                is_mention=self._user_id is not None and f"<@{self._user_id}>" in msg.get("text", ""),
                is_dm=channel_info.get("is_im", False),
            ))
        return messages

    def send_message(self, channel_id: str, text: str, thread_id: str | None = None) -> str:
        """Send a message via chat.postMessage. Returns message timestamp."""
        params: dict = {"channel": channel_id, "text": text}
        if thread_id:
            params["thread_ts"] = thread_id
        result = self._call("chat.postMessage", **params)
        return result.get("ts", "")

    def get_unread_mentions(self) -> list[SocialMessage]:
        """Get unread mentions by searching for messages directed at user."""
        if not self._user_id:
            return []
        return self.search_messages(f"<@{self._user_id}>", limit=20)

    def _call(self, method: str, **params) -> dict:
        """Make an authenticated Slack API call with rate-limit retry.

        All Slack API calls go through this helper.
        """
        if requests is None:
            raise ImportError("requests library required for Slack integration")
        if not self._token:
            raise RuntimeError("Not connected — call connect() first")

        url = f"{self.BASE_URL}/{method}"
        headers = {"Authorization": f"Bearer {self._token}"}

        max_retries = 3
        for attempt in range(max_retries):
            resp = requests.post(url, headers=headers, data=params, timeout=30)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "1"))
                logger.warning("Slack rate limited, retrying after %ds", retry_after)
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            return resp.json()

        # Final attempt after retries exhausted
        resp = requests.post(url, headers=headers, data=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
