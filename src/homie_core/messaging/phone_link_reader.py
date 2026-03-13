from __future__ import annotations
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_BASE = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Packages", "Microsoft.YourPhone_8wekyb3d8bbwe", "LocalCache", "Indexed",
)


class PhoneLinkReader:
    def __init__(self, base_path: str | None = None):
        self._base = Path(base_path) if base_path else Path(_DEFAULT_BASE).parent
        self._device_guid: str | None = None

    def is_available(self) -> bool:
        return len(self.discover_devices()) > 0

    def discover_devices(self) -> list[str]:
        indexed = self._base / "Indexed"
        if not indexed.exists():
            return []
        guids = []
        for entry in indexed.iterdir():
            if entry.is_dir() and (entry / "System" / "Database").exists():
                guids.append(entry.name)
        return guids

    def select_device(self, guid: str) -> None:
        self._device_guid = guid

    def read_messages(self, limit: int = 50) -> list[dict]:
        db_path = self._find_db()
        if not db_path:
            return []
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT ?", (limit,))
            messages = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return messages
        except Exception:
            logger.debug("Failed to read Phone Link messages", exc_info=True)
            return []

    def _find_db(self) -> Path | None:
        if not self._device_guid:
            devices = self.discover_devices()
            if not devices:
                return None
            self._device_guid = devices[0]
        db_dir = self._base / "Indexed" / self._device_guid / "System" / "Database"
        if db_dir.exists():
            for f in db_dir.iterdir():
                if f.suffix in (".db", ".sqlite", ".sqlite3") or f.name == "phone.db":
                    return f
        return None
