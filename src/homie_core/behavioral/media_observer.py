from __future__ import annotations

import logging
import platform
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

from homie_core.behavioral.base import BaseObserver


class MediaObserver(BaseObserver):
    def __init__(self):
        super().__init__(name="media")
        self._genre_counts: dict[str, int] = defaultdict(int)
        self._artist_counts: dict[str, int] = defaultdict(int)
        self._skip_count: int = 0
        self._play_count: int = 0
        self._last_track: str | None = None

    def tick(self) -> dict[str, Any]:
        info = self._get_now_playing()
        if info and info.get("title") != self._last_track:
            self._last_track = info.get("title")
            self._play_count += 1
            if info.get("artist"):
                self._artist_counts[info["artist"]] += 1
            self.record({"type": "track_change", **info})
        return info or {}

    def record_skip(self) -> None:
        self._skip_count += 1
        self.record({"type": "skip"})

    def get_profile_updates(self) -> dict[str, Any]:
        top_artists = sorted(self._artist_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return {
            "top_artists": [a[0] for a in top_artists],
            "play_count": self._play_count,
            "skip_count": self._skip_count,
            "skip_rate": self._skip_count / max(1, self._play_count),
        }

    def _get_now_playing(self) -> dict[str, Any] | None:
        if platform.system() == "Windows":
            try:
                import winsdk.windows.media.control as mc
                import asyncio
                return asyncio.run(self._get_windows_media())
            except (ImportError, Exception):
                return None
        return None

    async def _get_windows_media(self) -> dict[str, Any] | None:
        try:
            import winsdk.windows.media.control as mc
            manager = await mc.GlobalSystemMediaTransportControlsSessionManager.request_async()
            session = manager.get_current_session()
            if session:
                info = await session.try_get_media_properties_async()
                return {
                    "title": info.title or "",
                    "artist": info.artist or "",
                    "album": info.album_title or "",
                }
        except Exception as e:
            logger.debug("Windows media API query failed: %s", e)
        return None
