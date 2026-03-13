"""Windows Task Scheduler integration — registers Homie to start on login."""
from __future__ import annotations
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

_TASK_NAME = "HomieAI"


class ServiceManager:
    def __init__(self, task_name: str = _TASK_NAME):
        self._task_name = task_name
        self._python = sys.executable

    def register(self) -> bool:
        """Register Homie daemon to start on user login."""
        cmd = [
            "schtasks", "/create",
            "/tn", self._task_name,
            "/tr", f'"{self._python}" -m homie_app.cli daemon --headless',
            "/sc", "onlogon",
            "/rl", "limited",
            "/f",
        ]
        return self._run(cmd)

    def unregister(self) -> bool:
        cmd = ["schtasks", "/delete", "/tn", self._task_name, "/f"]
        return self._run(cmd)

    def is_registered(self) -> bool:
        try:
            result = subprocess.run(
                ["schtasks", "/query", "/tn", self._task_name, "/fo", "list"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def status(self) -> str:
        try:
            result = subprocess.run(
                ["schtasks", "/query", "/tn", self._task_name, "/fo", "list"],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout if result.returncode == 0 else "Not registered"
        except Exception:
            return "Unknown"

    def _run(self, cmd: list[str]) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning("Service command failed: %s", result.stderr)
                return False
            return True
        except Exception:
            logger.warning("Service command error", exc_info=True)
            return False
