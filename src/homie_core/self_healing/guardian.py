"""OS-level guardian process — monitors and restarts the watchdog if it dies."""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class Guardian:
    """Lightweight guardian that tracks the watchdog process via PID file."""

    def __init__(self, pid_file: Path | str) -> None:
        self._pid_file = Path(pid_file)

    def write_pid(self, pid: int) -> None:
        """Write the watchdog PID to the pid file."""
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(pid))

    def read_pid(self) -> Optional[int]:
        """Read the watchdog PID from the pid file."""
        if not self._pid_file.exists():
            return None
        try:
            return int(self._pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def is_alive(self) -> bool:
        """Check if the watchdog process is still running."""
        pid = self.read_pid()
        if pid is None:
            return False
        if psutil is None:
            # Fallback: try os.kill with signal 0
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        return psutil.pid_exists(pid)

    def cleanup(self) -> None:
        """Remove the PID file."""
        if self._pid_file.exists():
            self._pid_file.unlink()
