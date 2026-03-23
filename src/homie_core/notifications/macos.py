"""macOS notification backend using osascript."""
from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


class MacOSNotificationBackend:
    """Send desktop notifications on macOS via osascript."""

    def __init__(self, app_name: str = "Homie AI") -> None:
        self._app_name = app_name

    def is_available(self) -> bool:
        return shutil.which("osascript") is not None

    def send(self, title: str, message: str) -> bool:
        if not self.is_available():
            return False
        # Escape double-quotes for AppleScript string literals
        safe_title = title.replace('\\', '\\\\').replace('"', '\\"')
        safe_message = message.replace('\\', '\\\\').replace('"', '\\"')
        script = (
            f'display notification "{safe_message}" '
            f'with title "{safe_title}"'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                timeout=5,
                capture_output=True,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            logger.debug("osascript notification failed", exc_info=True)
            return False
