from unittest.mock import patch, MagicMock
from homie_app.service.tray_menu import build_tray_menu, toggle_dnd, toggle_screen_dnd


class TestTrayMenu:
    def test_build_menu_items(self):
        cfg = MagicMock()
        cfg.voice.enabled = True
        cfg.voice.mode = "push_to_talk"
        cfg.screen_reader.enabled = True
        cfg.screen_reader.level = 2
        cfg.screen_reader.dnd = False
        cfg.notifications.dnd_schedule_enabled = False

        items = build_tray_menu(cfg)
        labels = [item["label"] for item in items]
        assert "Open Chat" in labels
        assert "Settings..." in labels
        assert "Do Not Disturb" in labels
        assert "Pause Screen Reader" in labels
        assert "Stop Homie" in labels

    def test_dnd_toggle(self):
        cfg = MagicMock()
        cfg.notifications.dnd_schedule_enabled = False
        toggle_dnd(cfg)
        # After toggle, DND should be on
        assert cfg.notifications.dnd_schedule_enabled is True

    def test_screen_dnd_toggle(self):
        cfg = MagicMock()
        cfg.screen_reader.dnd = False
        toggle_screen_dnd(cfg)
        assert cfg.screen_reader.dnd is True
