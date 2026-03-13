from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from telethon import TelegramClient
    _HAS_TELETHON = True
except ImportError:
    _HAS_TELETHON = False


class TelegramProvider:
    def __init__(self, api_id: int, api_hash: str, session_dir: str = "~/.homie"):
        if not api_id or not api_hash:
            raise ValueError("api_id and api_hash are required")
        self._api_id = api_id
        self._api_hash = api_hash
        self._session_path = Path(session_dir).expanduser() / "telegram_session"
        self._client = None
        self.connected = False

    async def connect(self, phone: str, code_callback: Callable[[], str] | None = None) -> None:
        if not _HAS_TELETHON:
            raise ImportError("telethon is required: pip install telethon")
        self._client = TelegramClient(str(self._session_path), self._api_id, self._api_hash)
        await self._client.start(phone=phone, code_callback=code_callback)
        self.connected = True

    async def disconnect(self) -> None:
        if self._client:
            await self._client.disconnect()
            self.connected = False

    async def get_me(self) -> dict | None:
        if not self._client:
            return None
        me = await self._client.get_me()
        return {"id": me.id, "username": me.username, "phone": me.phone}

    async def get_recent_messages(self, limit: int = 20) -> list[dict]:
        if not self._client:
            return []
        messages = []
        async for dialog in self._client.iter_dialogs(limit=limit):
            messages.append({"chat": dialog.name, "last_message": dialog.message.text if dialog.message else "", "unread": dialog.unread_count})
        return messages

    async def send_message(self, chat: str | int, text: str) -> bool:
        if not self._client:
            return False
        try:
            await self._client.send_message(chat, text)
            return True
        except Exception:
            logger.warning("Failed to send Telegram message", exc_info=True)
            return False
