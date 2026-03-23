"""Windows toast notification backend."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_HAS_WINDOWS_TOASTS = False
try:
    from windows_toasts import Toast, InteractableWindowsToaster  # type: ignore[import-untyped]
    _HAS_WINDOWS_TOASTS = True
except ImportError:
    Toast = None  # type: ignore[assignment,misc]
    InteractableWindowsToaster = None  # type: ignore[assignment,misc]


class WindowsNotificationBackend:
    """Send native Windows toast notifications via windows-toasts."""

    def __init__(self, app_name: str = "Homie AI") -> None:
        self._app_name = app_name
        self._toaster = None
        if _HAS_WINDOWS_TOASTS and InteractableWindowsToaster is not None:
            try:
                self._toaster = InteractableWindowsToaster(app_name)
            except Exception:
                logger.debug("Failed to initialise Windows toaster", exc_info=True)

    def is_available(self) -> bool:
        return self._toaster is not None

    def send(self, title: str, message: str) -> bool:
        if not self.is_available():
            return False
        try:
            toast = Toast()
            toast.text_fields = [title, message]
            self._toaster.show_toast(toast)  # type: ignore[union-attr]
            return True
        except Exception:
            logger.debug("Windows toast failed", exc_info=True)
            return False
