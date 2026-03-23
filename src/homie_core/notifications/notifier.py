"""Cross-platform notification dispatcher with DND support."""
from __future__ import annotations

import logging
import sys
from datetime import datetime

from homie_core.config import NotificationConfig

logger = logging.getLogger(__name__)


class _ConsoleBackend:
    """Fallback backend that logs notifications to the console."""

    def is_available(self) -> bool:
        return True

    def send(self, title: str, message: str) -> bool:
        logger.info("[Notification] %s: %s", title, message)
        return True


def _create_platform_backend(app_name: str = "Homie AI"):
    """Auto-detect platform and return the appropriate backend.

    Returns the native backend if available, otherwise the console fallback.
    """
    platform = sys.platform

    if platform == "win32":
        from homie_core.notifications.windows import WindowsNotificationBackend
        backend = WindowsNotificationBackend(app_name=app_name)
        if backend.is_available():
            return backend

    elif platform == "darwin":
        from homie_core.notifications.macos import MacOSNotificationBackend
        backend = MacOSNotificationBackend(app_name=app_name)
        if backend.is_available():
            return backend

    elif platform.startswith("linux"):
        from homie_core.notifications.linux import LinuxNotificationBackend
        backend = LinuxNotificationBackend(app_name=app_name)
        if backend.is_available():
            return backend

    logger.debug(
        "No native notification backend for platform %r; falling back to console",
        platform,
    )
    return _ConsoleBackend()


class Notifier:
    """Cross-platform notification dispatcher.

    Automatically selects the right backend for the current OS and respects
    the DND (Do Not Disturb) schedule from :class:`NotificationConfig`.

    Parameters
    ----------
    config:
        Notification configuration (enabled flag, categories, DND schedule).
    app_name:
        Application name shown in native notifications.
    backend:
        Override the auto-detected backend (useful for testing).
    """

    def __init__(
        self,
        config: NotificationConfig | None = None,
        app_name: str = "Homie AI",
        backend=None,
    ) -> None:
        self._config = config or NotificationConfig()
        self._backend = backend or _create_platform_backend(app_name)

    # -- public API ----------------------------------------------------------

    def notify(
        self,
        title: str,
        message: str,
        priority: str = "normal",
    ) -> bool:
        """Send a notification.

        Returns ``True`` if the notification was delivered, ``False`` if it was
        suppressed (disabled, DND) or delivery failed.

        Priority levels: ``"low"``, ``"normal"``, ``"high"``, ``"critical"``.
        Only ``"critical"`` bypasses DND.
        """
        if not self._config.enabled:
            logger.debug("Notifications disabled globally")
            return False

        if priority != "critical" and self._is_in_dnd():
            logger.debug("Notification suppressed by DND: %s", title)
            return False

        return self._backend.send(title, message)

    def is_available(self) -> bool:
        """Return whether the underlying backend can deliver notifications."""
        return self._backend.is_available()

    # -- DND helpers ---------------------------------------------------------

    def _is_in_dnd(self, current_time: str | None = None) -> bool:
        """Check if the current time falls within the DND schedule."""
        if not self._config.dnd_schedule_enabled:
            return False

        now = current_time or datetime.now().strftime("%H:%M")
        start = self._config.dnd_schedule_start
        end = self._config.dnd_schedule_end

        if start <= end:
            return start <= now <= end
        else:
            # Wraps midnight, e.g. 22:00 -> 07:00
            return now >= start or now < end
