from unittest.mock import patch, MagicMock
from homie_core.notifications.toast import ToastNotifier


class TestToastNotifier:
    def test_init(self):
        notifier = ToastNotifier()
        assert notifier is not None

    @patch("homie_core.notifications.toast._HAS_WINDOWS_TOASTS", True)
    @patch("homie_core.notifications.toast.Toast")
    @patch("homie_core.notifications.toast.InteractableWindowsToaster")
    def test_show_toast(self, mock_toaster_cls, mock_toast_cls):
        mock_toaster = MagicMock()
        mock_toaster_cls.return_value = mock_toaster
        notifier = ToastNotifier()
        notifier.show("Test Title", "Test Body")
        assert mock_toaster.show_toast.called

    @patch("homie_core.notifications.toast._HAS_WINDOWS_TOASTS", False)
    @patch("homie_core.notifications.toast._HAS_PLYER", True)
    @patch("homie_core.notifications.toast.plyer_notification")
    def test_fallback_to_plyer(self, mock_plyer):
        notifier = ToastNotifier()
        notifier.show("Test Title", "Test Body")
        mock_plyer.notify.assert_called_once()

    @patch("homie_core.notifications.toast._HAS_WINDOWS_TOASTS", False)
    @patch("homie_core.notifications.toast._HAS_PLYER", False)
    def test_no_library_logs_warning(self):
        notifier = ToastNotifier()
        # Should not raise, just log
        notifier.show("Test Title", "Test Body")
