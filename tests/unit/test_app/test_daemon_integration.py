from unittest.mock import patch, MagicMock
from homie_core.config import HomieConfig


class TestDaemonScreenReader:
    @patch("homie_core.screen_reader.capture_scheduler.CaptureScheduler")
    def test_screen_reader_initialized_when_enabled(self, mock_sched_cls):
        cfg = HomieConfig()
        cfg.screen_reader.enabled = True
        cfg.screen_reader.level = 1
        from homie_app.daemon import HomieDaemon
        with patch.object(HomieDaemon, "__init__", lambda self, **kw: None):
            d = HomieDaemon.__new__(HomieDaemon)
            d._config = cfg
            d._init_screen_reader()
            mock_sched_cls.assert_called_once()

    def test_screen_reader_not_initialized_when_disabled(self):
        cfg = HomieConfig()
        cfg.screen_reader.enabled = False
        from homie_app.daemon import HomieDaemon
        with patch.object(HomieDaemon, "__init__", lambda self, **kw: None):
            d = HomieDaemon.__new__(HomieDaemon)
            d._config = cfg
            d._screen_scheduler = None
            d._init_screen_reader()
            assert d._screen_scheduler is None


class TestDaemonNotifications:
    @patch("homie_core.notifications.router.NotificationRouter")
    @patch("homie_core.notifications.toast.ToastNotifier")
    def test_notification_system_initialized(self, mock_toast, mock_router):
        cfg = HomieConfig()
        cfg.notifications.enabled = True
        from homie_app.daemon import HomieDaemon
        with patch.object(HomieDaemon, "__init__", lambda self, **kw: None):
            d = HomieDaemon.__new__(HomieDaemon)
            d._config = cfg
            d._init_notifications()
            mock_router.assert_called_once()
            mock_toast.assert_called_once()
