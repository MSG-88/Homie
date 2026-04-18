from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
from homie_core.mesh.events import generate_ulid
from homie_core.utils import utc_now

class TaskState:
    PENDING = "pending"; RUNNING = "running"; COMPLETED = "completed"
    FAILED = "failed"; REJECTED = "rejected"

_DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\s+(-[a-z]*r[a-z]*f|--force|-[a-z]*f[a-z]*r)", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+(table|database)\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+from\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bgit\s+push\s+--force\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bkill\s+-9\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
]

@dataclass
class MeshTask:
    source_node: str; target_node: str; command: str; reason: str = ""
    task_id: str = field(default_factory=generate_ulid)
    state: str = field(default=TaskState.PENDING)
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    started_at: str = ""; finished_at: str = ""
    stdout: str = ""; stderr: str = ""
    exit_code: Optional[int] = None; error: Optional[str] = None
    dry_run: bool = False; requires_confirmation: bool = False

    @property
    def is_destructive(self) -> bool:
        return any(p.search(self.command) for p in _DESTRUCTIVE_PATTERNS)

    def complete(self, stdout="", stderr="", exit_code=0):
        self.state = TaskState.COMPLETED; self.stdout = stdout; self.stderr = stderr
        self.exit_code = exit_code; self.finished_at = utc_now().isoformat()

    def fail(self, error: str, exit_code=None):
        self.state = TaskState.FAILED; self.error = error
        self.exit_code = exit_code; self.finished_at = utc_now().isoformat()

    def reject(self, reason: str):
        self.state = TaskState.REJECTED; self.error = reason
        self.finished_at = utc_now().isoformat()

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "source_node": self.source_node,
                "target_node": self.target_node, "command": self.command,
                "reason": self.reason, "state": self.state, "created_at": self.created_at,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "stdout": self.stdout, "stderr": self.stderr, "exit_code": self.exit_code,
                "error": self.error, "dry_run": self.dry_run,
                "requires_confirmation": self.requires_confirmation}

    @classmethod
    def from_dict(cls, d: dict) -> MeshTask:
        return cls(task_id=d["task_id"], source_node=d["source_node"],
                   target_node=d["target_node"], command=d["command"],
                   reason=d.get("reason", ""), state=d.get("state", TaskState.PENDING),
                   created_at=d.get("created_at", ""), started_at=d.get("started_at", ""),
                   finished_at=d.get("finished_at", ""), stdout=d.get("stdout", ""),
                   stderr=d.get("stderr", ""), exit_code=d.get("exit_code"),
                   error=d.get("error"), dry_run=d.get("dry_run", False),
                   requires_confirmation=d.get("requires_confirmation", False))
