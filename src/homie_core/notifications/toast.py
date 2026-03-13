from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

Toast = None
InteractableWindowsToaster = None
ToastText1 = None
try:
    from windows_toasts import Toast, InteractableWindowsToaster, ToastText1
    _HAS_WINDOWS_TOASTS = True
except ImportError:
    _HAS_WINDOWS_TOASTS = False

plyer_notification = None
try:
    from plyer import notification as plyer_notification
    _HAS_PLYER = True
except ImportError:
    _HAS_PLYER = False


class ToastNotifier:
    def __init__(self, app_name: str = "Homie AI"):
        self._app_name = app_name
        self._toaster = None
        if _HAS_WINDOWS_TOASTS:
            self._toaster = InteractableWindowsToaster(app_name)

    def show(self, title: str, body: str) -> None:
        if _HAS_WINDOWS_TOASTS and self._toaster:
            self._show_windows_toast(title, body)
        elif _HAS_PLYER:
            self._show_plyer(title, body)
        else:
            logger.warning("No notification library available. Install windows-toasts or plyer.")

    def _show_windows_toast(self, title: str, body: str) -> None:
        try:
            toast = Toast()
            toast.text_fields = [title, body]
            self._toaster.show_toast(toast)
        except Exception:
            logger.debug("Windows toast failed", exc_info=True)

    def _show_plyer(self, title: str, body: str) -> None:
        try:
            plyer_notification.notify(
                title=title,
                message=body,
                app_name=self._app_name,
                timeout=10,
            )
        except Exception:
            logger.debug("Plyer notification failed", exc_info=True)
