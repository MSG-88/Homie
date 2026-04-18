from __future__ import annotations
import logging, os, subprocess, threading
from typing import Optional
from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.utils import utc_now

logger = logging.getLogger(__name__)


class MeshTaskExecutor:
    def __init__(self, allowlist: Optional[list[str]] = None, timeout_seconds: int = 120):
        self._allowlist = set(allowlist or [])
        self._timeout = timeout_seconds

    def execute(self, task: MeshTask) -> None:
        if task.is_destructive and task.command not in self._allowlist:
            task.reject("Destructive command blocked by safety gate")
            return
        if task.dry_run:
            task.complete(stdout="", exit_code=0)
            return
        task.state = TaskState.RUNNING
        task.started_at = utc_now().isoformat()
        try:
            proc = subprocess.Popen(task.command, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, env=os.environ.copy())
            timer = threading.Timer(self._timeout, proc.kill)
            timer.start()
            try:
                stdout, stderr = proc.communicate(timeout=self._timeout + 1)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                task.fail(error=f"Timeout after {self._timeout}s", exit_code=-1)
                return
            finally:
                timer.cancel()
            if proc.returncode == 0:
                task.complete(stdout=stdout, stderr=stderr, exit_code=0)
            else:
                task.fail(error=stderr.strip() or f"Exit code {proc.returncode}", exit_code=proc.returncode)
        except Exception as e:
            task.fail(error=str(e))
