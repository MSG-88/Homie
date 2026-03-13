from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def build_tray_menu(cfg) -> list[dict]:
    """Build tray menu items list for pystray integration."""
    voice_status = f"Voice: {cfg.voice.mode}" if cfg.voice.enabled else "Voice: disabled"
    screen_status = f"Screen: level {cfg.screen_reader.level}" if cfg.screen_reader.enabled else "Screen: off"

    return [
        {"label": "Open Chat", "action": "open_chat"},
        {"label": "Settings...", "action": "open_settings"},
        {"label": "separator"},
        {"label": "Status: Running", "action": None},
        {"label": voice_status, "action": None},
        {"label": screen_status, "action": None},
        {"label": "separator"},
        {"label": "Do Not Disturb", "action": "toggle_dnd", "checked": cfg.notifications.dnd_schedule_enabled},
        {"label": "Pause Screen Reader", "action": "toggle_screen_dnd", "checked": cfg.screen_reader.dnd},
        {"label": "separator"},
        {"label": "Stop Homie", "action": "stop"},
    ]


def toggle_dnd(cfg) -> None:
    cfg.notifications.dnd_schedule_enabled = not cfg.notifications.dnd_schedule_enabled


def toggle_screen_dnd(cfg) -> None:
    cfg.screen_reader.dnd = not cfg.screen_reader.dnd
