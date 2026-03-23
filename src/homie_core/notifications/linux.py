"""Linux notification backend using notify-send."""
from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


class LinuxNotificationBackend:
    """Send desktop notifications on Linux via notify-send."""

    def __init__(self, app_name: str = "Homie AI") -> None:
        self._app_name = app_name

    def is_available(self) -> bool:
        return shutil.which("notify-send") is not None

    def send(self, title: str, message: str) -> bool:
        if not self.is_available():
            return False
        try:
            subprocess.run(
                ["notify-send", "--app-name", self._app_name, title, message],
                check=True,
                timeout=5,
                capture_output=True,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            logger.debug("notify-send failed", exc_info=True)
            return False
