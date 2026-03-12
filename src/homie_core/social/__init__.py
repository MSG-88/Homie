"""Social/messaging integration — provider abstraction, sync, and tools.

SocialService is the main facade used by the daemon and CLI.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from homie_core.social.models import SocialChannel, SocialMessage, SocialNotification
from homie_core.social.provider import SocialProvider

logger = logging.getLogger(__name__)


class SocialService:
    """High-level facade for social/messaging operations.

    Used by the daemon for sync callbacks and by tools for queries.
    """

    def __init__(self, vault, working_memory=None):
        self._vault = vault
        self._working_memory = working_memory
        self._providers: dict[str, SocialProvider] = {}  # platform -> provider

    def initialize(self) -> list[str]:
        """Initialize providers from stored credentials.

        Returns list of connected platform names.
        """
        from homie_core.social.slack_provider import SlackProvider

        connected = []

        # --- Slack ---
        slack_creds = self._vault.list_credentials("slack")
        for cred in slack_creds:
            if not cred.active:
                continue
            if cred.account_id == "oauth_client":
                continue
            try:
                provider = SlackProvider(account_id=cred.account_id)
                if provider.connect(cred):
                    self._providers["slack"] = provider
                    self._vault.set_connection_status(
                        "slack", connected=True, label=cred.account_id,
                    )
                    connected.append("slack")
                    break  # One Slack workspace at a time
            except Exception:
                logger.exception("Failed to connect Slack account %s", cred.account_id)

        return connected

    def sync_tick(self) -> str:
        """Called by SyncManager — check for new mentions/DMs across platforms."""
        parts = []
        for platform, provider in self._providers.items():
            try:
                mentions = provider.get_unread_mentions()
                if mentions and self._working_memory is not None:
                    for msg in mentions[:5]:
                        notification = SocialNotification(
                            message=msg,
                            reason="mention" if msg.is_mention else "dm",
                        )
                        self._working_memory.push(
                            f"[{platform}] New {notification.reason} from {msg.sender}: {msg.content[:100]}"
                        )
                count = len(mentions)
                parts.append(f"{platform}: {count} mention(s)")
            except Exception as exc:
                parts.append(f"{platform}: error ({exc})")
        return "; ".join(parts) if parts else "No social platforms connected"

    def list_channels(self, platform: str = "all") -> list[dict]:
        """List channels across platforms."""
        channels: list[dict] = []
        for plat, provider in self._providers.items():
            if platform != "all" and plat != platform:
                continue
            try:
                for ch in provider.list_channels():
                    channels.append(ch.to_dict())
            except Exception:
                pass
        return channels

    def get_messages(self, channel_id: str, platform: str | None = None, limit: int = 20) -> list[dict]:
        """Get recent messages from a channel."""
        for plat, provider in self._providers.items():
            if platform and plat != platform:
                continue
            try:
                messages = provider.get_recent_messages(channel_id, limit=limit)
                return [m.to_dict() for m in messages]
            except Exception:
                continue
        return []

    def search(self, query: str, platform: str = "all", limit: int = 10) -> list[dict]:
        """Search messages across platforms."""
        results: list[dict] = []
        for plat, provider in self._providers.items():
            if platform != "all" and plat != platform:
                continue
            try:
                messages = provider.search_messages(query, limit=limit)
                results.extend(m.to_dict() for m in messages)
            except Exception:
                pass
        return results[:limit]

    def send_message(self, channel_id: str, text: str, platform: str | None = None,
                     thread_id: str | None = None) -> dict:
        """Send a message to a channel."""
        for plat, provider in self._providers.items():
            if platform and plat != platform:
                continue
            try:
                msg_id = provider.send_message(channel_id, text, thread_id=thread_id)
                return {"status": "sent", "platform": plat, "message_id": msg_id}
            except Exception as exc:
                return {"status": "error", "platform": plat, "error": str(exc)}
        return {"status": "error", "error": "No matching provider"}

    def get_unread(self) -> dict:
        """Get unread mentions/DMs grouped by platform."""
        grouped: dict[str, list[dict]] = {}
        for plat, provider in self._providers.items():
            try:
                mentions = provider.get_unread_mentions()
                grouped[plat] = [m.to_dict() for m in mentions]
            except Exception:
                grouped[plat] = []
        return grouped
