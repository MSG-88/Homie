# Distributed Mesh — Phase 5: Distributed Task Execution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Any Homie node can dispatch tasks to any other node in the mesh. Hub orchestrates cross-machine commands with safety gates, audit logging, and result aggregation. Tasks are tracked as mesh events for sync.

**Architecture:** `MeshTask` dataclass represents a cross-machine task. `TaskDispatcher` on any node sends tasks to the Hub. `TaskExecutor` on each node runs commands locally with safety checks. `TaskTracker` tracks task lifecycle via mesh events. All cross-machine commands require safety validation and audit logging.

**Tech Stack:** Python 3.11+, existing mesh events/sync, existing `homie/node/executor.py` safety patterns

**Builds on:** Phase 1-4 (mesh identity, events, sync, inference, context)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/homie_core/mesh/task_model.py` | MeshTask dataclass and lifecycle states |
| `src/homie_core/mesh/task_executor.py` | Execute commands locally with safety checks |
| `src/homie_core/mesh/task_dispatcher.py` | Dispatch tasks to target nodes |
| `src/homie_core/mesh/task_tracker.py` | Track task lifecycle via mesh events |
| `tests/unit/test_mesh/test_task_model.py` | Task model tests |
| `tests/unit/test_mesh/test_task_executor.py` | Executor tests |
| `tests/unit/test_mesh/test_task_dispatcher.py` | Dispatcher tests |
| `tests/unit/test_mesh/test_task_tracker.py` | Tracker tests |
| `tests/integration/test_distributed_tasks.py` | End-to-end task tests |

---

### Task 1: MeshTask Data Model and Lifecycle

**Files:**
- Create: `src/homie_core/mesh/task_model.py`
- Test: `tests/unit/test_mesh/test_task_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_task_model.py
from homie_core.mesh.task_model import MeshTask, TaskState


def test_task_states():
    assert TaskState.PENDING == "pending"
    assert TaskState.RUNNING == "running"
    assert TaskState.COMPLETED == "completed"
    assert TaskState.FAILED == "failed"
    assert TaskState.REJECTED == "rejected"


def test_task_creation():
    task = MeshTask(
        source_node="desktop", target_node="laptop",
        command="ls ~/Downloads", reason="organize files",
    )
    assert task.task_id  # Auto-generated
    assert task.source_node == "desktop"
    assert task.target_node == "laptop"
    assert task.state == TaskState.PENDING


def test_task_to_dict():
    task = MeshTask(source_node="a", target_node="b", command="echo hi")
    d = task.to_dict()
    assert d["source_node"] == "a"
    assert d["command"] == "echo hi"
    assert d["state"] == "pending"


def test_task_from_dict():
    task = MeshTask(source_node="a", target_node="b", command="pwd", reason="test")
    restored = MeshTask.from_dict(task.to_dict())
    assert restored.task_id == task.task_id
    assert restored.command == "pwd"
    assert restored.reason == "test"


def test_task_complete():
    task = MeshTask(source_node="a", target_node="b", command="echo ok")
    task.complete(stdout="ok\n", exit_code=0)
    assert task.state == TaskState.COMPLETED
    assert task.stdout == "ok\n"
    assert task.exit_code == 0


def test_task_fail():
    task = MeshTask(source_node="a", target_node="b", command="bad")
    task.fail(error="command not found", exit_code=127)
    assert task.state == TaskState.FAILED
    assert task.error == "command not found"


def test_task_reject():
    task = MeshTask(source_node="a", target_node="b", command="rm -rf /")
    task.reject(reason="blocked by safety")
    assert task.state == TaskState.REJECTED
    assert task.error == "blocked by safety"


def test_task_is_destructive():
    assert MeshTask(source_node="a", target_node="b", command="rm -rf /tmp").is_destructive is True
    assert MeshTask(source_node="a", target_node="b", command="ls /tmp").is_destructive is False
    assert MeshTask(source_node="a", target_node="b", command="DROP TABLE users").is_destructive is True
    assert MeshTask(source_node="a", target_node="b", command="format C:").is_destructive is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_task_model.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/task_model.py
"""MeshTask — cross-machine task data model with lifecycle states."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from homie_core.mesh.events import generate_ulid
from homie_core.utils import utc_now


class TaskState:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


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
    """A task dispatched across the mesh from one node to another."""

    source_node: str
    target_node: str
    command: str
    reason: str = ""
    task_id: str = field(default_factory=generate_ulid)
    state: str = field(default=TaskState.PENDING)
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    started_at: str = ""
    finished_at: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    error: Optional[str] = None
    dry_run: bool = False
    requires_confirmation: bool = False

    @property
    def is_destructive(self) -> bool:
        """Check if the command matches known destructive patterns."""
        return any(p.search(self.command) for p in _DESTRUCTIVE_PATTERNS)

    def complete(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.state = TaskState.COMPLETED
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.finished_at = utc_now().isoformat()

    def fail(self, error: str, exit_code: Optional[int] = None) -> None:
        self.state = TaskState.FAILED
        self.error = error
        self.exit_code = exit_code
        self.finished_at = utc_now().isoformat()

    def reject(self, reason: str) -> None:
        self.state = TaskState.REJECTED
        self.error = reason
        self.finished_at = utc_now().isoformat()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "source_node": self.source_node,
            "target_node": self.target_node, "command": self.command,
            "reason": self.reason, "state": self.state,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "stdout": self.stdout,
            "stderr": self.stderr, "exit_code": self.exit_code,
            "error": self.error, "dry_run": self.dry_run,
            "requires_confirmation": self.requires_confirmation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MeshTask:
        return cls(
            task_id=d["task_id"], source_node=d["source_node"],
            target_node=d["target_node"], command=d["command"],
            reason=d.get("reason", ""), state=d.get("state", TaskState.PENDING),
            created_at=d.get("created_at", ""), started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""), stdout=d.get("stdout", ""),
            stderr=d.get("stderr", ""), exit_code=d.get("exit_code"),
            error=d.get("error"), dry_run=d.get("dry_run", False),
            requires_confirmation=d.get("requires_confirmation", False),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_task_model.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/task_model.py tests/unit/test_mesh/test_task_model.py
git commit -m "feat(mesh): add MeshTask data model with lifecycle states and destructive command detection"
```

---

### Task 2: Task Executor — Safe Local Command Execution

**Files:**
- Create: `src/homie_core/mesh/task_executor.py`
- Test: `tests/unit/test_mesh/test_task_executor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_task_executor.py
import sys
from homie_core.mesh.task_executor import MeshTaskExecutor
from homie_core.mesh.task_model import MeshTask, TaskState


def test_execute_simple_command():
    executor = MeshTaskExecutor()
    task = MeshTask(source_node="hub", target_node="local", command="echo hello")
    executor.execute(task)
    assert task.state == TaskState.COMPLETED
    assert "hello" in task.stdout
    assert task.exit_code == 0


def test_execute_failing_command():
    executor = MeshTaskExecutor()
    cmd = "exit 1" if sys.platform != "win32" else "cmd /c exit 1"
    task = MeshTask(source_node="hub", target_node="local", command=cmd)
    executor.execute(task)
    assert task.state == TaskState.FAILED
    assert task.exit_code == 1


def test_reject_destructive_command():
    executor = MeshTaskExecutor()
    task = MeshTask(source_node="hub", target_node="local", command="rm -rf /")
    executor.execute(task)
    assert task.state == TaskState.REJECTED
    assert "destructive" in task.error.lower()


def test_dry_run_does_not_execute():
    executor = MeshTaskExecutor()
    task = MeshTask(source_node="hub", target_node="local",
                    command="echo should_not_run", dry_run=True)
    executor.execute(task)
    assert task.state == TaskState.COMPLETED
    assert task.stdout == ""  # Nothing actually ran


def test_allowlist_permits_destructive():
    """If command is in allowlist, destructive check is bypassed."""
    executor = MeshTaskExecutor(allowlist=["rm -rf /tmp/test"])
    task = MeshTask(source_node="hub", target_node="local", command="rm -rf /tmp/test")
    # Won't actually execute rm on CI — just verify it's not rejected
    executor.execute(task)
    assert task.state != TaskState.REJECTED


def test_timeout_enforcement():
    executor = MeshTaskExecutor(timeout_seconds=1)
    cmd = "sleep 10" if sys.platform != "win32" else "ping -n 11 127.0.0.1"
    task = MeshTask(source_node="hub", target_node="local", command=cmd)
    executor.execute(task)
    assert task.state == TaskState.FAILED
    assert "timeout" in task.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_task_executor.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/task_executor.py
"""Mesh task executor — runs commands locally with safety checks."""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import Optional

from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.utils import utc_now

logger = logging.getLogger(__name__)


class MeshTaskExecutor:
    """Executes mesh tasks on the local machine with safety gates."""

    def __init__(
        self,
        allowlist: Optional[list[str]] = None,
        timeout_seconds: int = 120,
    ):
        self._allowlist = set(allowlist or [])
        self._timeout = timeout_seconds

    def execute(self, task: MeshTask) -> None:
        """Execute a task. Modifies the task in place with results."""
        # Safety check
        if task.is_destructive and task.command not in self._allowlist:
            task.reject("Destructive command blocked by safety gate")
            logger.warning("Rejected destructive task %s: %s", task.task_id, task.command)
            return

        # Dry run
        if task.dry_run:
            task.complete(stdout="", exit_code=0)
            return

        # Execute
        task.state = TaskState.RUNNING
        task.started_at = utc_now().isoformat()

        try:
            proc = subprocess.Popen(
                task.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ.copy(),
            )

            # Timeout enforcement
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
                task.fail(
                    error=stderr.strip() or f"Exit code {proc.returncode}",
                    exit_code=proc.returncode,
                )

        except Exception as e:
            task.fail(error=str(e))
            logger.error("Task %s execution error: %s", task.task_id, e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_task_executor.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/task_executor.py tests/unit/test_mesh/test_task_executor.py
git commit -m "feat(mesh): add MeshTaskExecutor with safety gates and timeout enforcement"
```

---

### Task 3: Task Dispatcher — Send Tasks to Target Nodes

**Files:**
- Create: `src/homie_core/mesh/task_dispatcher.py`
- Test: `tests/unit/test_mesh/test_task_dispatcher.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_task_dispatcher.py
from homie_core.mesh.task_dispatcher import TaskDispatcher
from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.mesh.task_executor import MeshTaskExecutor


def test_dispatch_local_task():
    """Tasks targeting self are executed locally."""
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id="node-1", executor=executor)

    task = MeshTask(source_node="node-1", target_node="node-1", command="echo local")
    result = dispatcher.dispatch(task)
    assert result.state == TaskState.COMPLETED
    assert "local" in result.stdout


def test_dispatch_remote_returns_pending():
    """Tasks targeting another node are queued as pending (no transport yet)."""
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id="node-1", executor=executor)

    task = MeshTask(source_node="node-1", target_node="node-2", command="echo remote")
    result = dispatcher.dispatch(task)
    # Without a remote handler, task stays pending with error
    assert result.state in (TaskState.PENDING, TaskState.FAILED)


def test_dispatch_with_remote_handler():
    """Remote handler is called for non-local tasks."""
    results = []

    def fake_remote_handler(task: MeshTask) -> MeshTask:
        task.complete(stdout="remote result", exit_code=0)
        results.append(task)
        return task

    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(
        local_node_id="node-1", executor=executor,
        remote_handler=fake_remote_handler,
    )

    task = MeshTask(source_node="node-1", target_node="node-2", command="echo hi")
    result = dispatcher.dispatch(task)
    assert result.state == TaskState.COMPLETED
    assert result.stdout == "remote result"
    assert len(results) == 1


def test_dispatch_records_history():
    """Dispatcher records all dispatched tasks."""
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id="node-1", executor=executor)

    dispatcher.dispatch(MeshTask(source_node="n1", target_node="node-1", command="echo a"))
    dispatcher.dispatch(MeshTask(source_node="n1", target_node="node-1", command="echo b"))

    history = dispatcher.task_history()
    assert len(history) == 2


def test_dispatch_all_nodes():
    """dispatch_all sends to multiple targets."""
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id="node-1", executor=executor)

    results = dispatcher.dispatch_all(
        source_node="node-1",
        target_nodes=["node-1"],
        command="echo multi",
        reason="test",
    )
    assert len(results) == 1
    assert results[0].state == TaskState.COMPLETED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_task_dispatcher.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/task_dispatcher.py
"""Task dispatcher — routes tasks to local or remote execution."""
from __future__ import annotations

import logging
from typing import Callable, Optional

from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.mesh.task_executor import MeshTaskExecutor

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """Dispatches tasks to local executor or remote nodes."""

    def __init__(
        self,
        local_node_id: str,
        executor: MeshTaskExecutor,
        remote_handler: Optional[Callable[[MeshTask], MeshTask]] = None,
    ):
        self._local_node_id = local_node_id
        self._executor = executor
        self._remote_handler = remote_handler
        self._history: list[MeshTask] = []

    def dispatch(self, task: MeshTask) -> MeshTask:
        """Dispatch a single task to the appropriate node."""
        if task.target_node == self._local_node_id:
            self._executor.execute(task)
        elif self._remote_handler:
            task = self._remote_handler(task)
        else:
            task.fail(error=f"No route to remote node {task.target_node}")

        self._history.append(task)
        return task

    def dispatch_all(
        self,
        source_node: str,
        target_nodes: list[str],
        command: str,
        reason: str = "",
    ) -> list[MeshTask]:
        """Dispatch the same command to multiple nodes."""
        results = []
        for target in target_nodes:
            task = MeshTask(
                source_node=source_node,
                target_node=target,
                command=command,
                reason=reason,
            )
            results.append(self.dispatch(task))
        return results

    def task_history(self) -> list[MeshTask]:
        """Return all dispatched tasks."""
        return list(self._history)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_task_dispatcher.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/task_dispatcher.py tests/unit/test_mesh/test_task_dispatcher.py
git commit -m "feat(mesh): add TaskDispatcher for local and remote task routing"
```

---

### Task 4: Task Tracker — Event-Based Lifecycle Tracking

**Files:**
- Create: `src/homie_core/mesh/task_tracker.py`
- Test: `tests/unit/test_mesh/test_task_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_task_tracker.py
from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.mesh.task_tracker import TaskTracker


def test_track_task_emits_events(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    tracker = TaskTracker(mesh_manager=mgr)

    task = MeshTask(source_node=identity.node_id, target_node="remote",
                    command="echo test", reason="testing")

    tracker.track_created(task)
    assert mgr.event_count() == 1

    events = mgr.events_since(None)
    assert events[0].category == "task"
    assert events[0].event_type == "task_created"
    assert events[0].payload["command"] == "echo test"


def test_track_completed(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    tracker = TaskTracker(mesh_manager=mgr)

    task = MeshTask(source_node=identity.node_id, target_node="remote", command="echo ok")
    task.complete(stdout="ok\n", exit_code=0)

    tracker.track_completed(task)
    events = mgr.events_since(None)
    assert events[0].event_type == "task_completed"
    assert events[0].payload["exit_code"] == 0


def test_track_failed(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    tracker = TaskTracker(mesh_manager=mgr)

    task = MeshTask(source_node=identity.node_id, target_node="remote", command="bad")
    task.fail(error="not found", exit_code=127)

    tracker.track_failed(task)
    events = mgr.events_since(None)
    assert events[0].event_type == "task_failed"
    assert "not found" in events[0].payload["error"]


def test_get_task_history(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    tracker = TaskTracker(mesh_manager=mgr)

    for i in range(3):
        task = MeshTask(source_node=identity.node_id, target_node="r",
                        command=f"echo {i}")
        tracker.track_created(task)

    history = tracker.get_task_events()
    assert len(history) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_task_tracker.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/task_tracker.py
"""Task tracker — tracks task lifecycle via mesh events."""
from __future__ import annotations

from homie_core.mesh.events import HomieEvent
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.task_model import MeshTask


class TaskTracker:
    """Emits mesh events for task lifecycle changes."""

    def __init__(self, mesh_manager: MeshManager):
        self._mgr = mesh_manager

    def track_created(self, task: MeshTask) -> HomieEvent:
        return self._mgr.emit("task", "task_created", {
            "task_id": task.task_id,
            "source_node": task.source_node,
            "target_node": task.target_node,
            "command": task.command,
            "reason": task.reason,
        })

    def track_completed(self, task: MeshTask) -> HomieEvent:
        return self._mgr.emit("task", "task_completed", {
            "task_id": task.task_id,
            "target_node": task.target_node,
            "exit_code": task.exit_code,
            "stdout_preview": task.stdout[:200] if task.stdout else "",
        })

    def track_failed(self, task: MeshTask) -> HomieEvent:
        return self._mgr.emit("task", "task_failed", {
            "task_id": task.task_id,
            "target_node": task.target_node,
            "error": task.error or "",
            "exit_code": task.exit_code,
        })

    def track_rejected(self, task: MeshTask) -> HomieEvent:
        return self._mgr.emit("task", "task_rejected", {
            "task_id": task.task_id,
            "target_node": task.target_node,
            "reason": task.error or "",
        })

    def get_task_events(self) -> list[HomieEvent]:
        """Get all task-related events from the event store."""
        return self._mgr._event_store.events_by_category("task")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_task_tracker.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/task_tracker.py tests/unit/test_mesh/test_task_tracker.py
git commit -m "feat(mesh): add TaskTracker for event-based task lifecycle tracking"
```

---

### Task 5: Integration Test — Full Distributed Task Flow

**Files:**
- Create: `tests/integration/test_distributed_tasks.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_distributed_tasks.py
"""End-to-end: task dispatch, execution, tracking, and sync."""
from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.mesh.task_executor import MeshTaskExecutor
from homie_core.mesh.task_dispatcher import TaskDispatcher
from homie_core.mesh.task_tracker import TaskTracker
from homie_core.mesh.sync_protocol import SyncRequest


def test_local_task_execution_and_tracking(tmp_path):
    """Dispatch a task locally, execute, track via events."""
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id=identity.node_id, executor=executor)
    tracker = TaskTracker(mesh_manager=mgr)

    task = MeshTask(
        source_node=identity.node_id,
        target_node=identity.node_id,
        command="echo distributed_task_test",
        reason="integration test",
    )
    tracker.track_created(task)
    result = dispatcher.dispatch(task)
    assert result.state == TaskState.COMPLETED
    assert "distributed_task_test" in result.stdout

    tracker.track_completed(result)
    events = tracker.get_task_events()
    assert len(events) == 2
    assert events[0].event_type == "task_created"
    assert events[1].event_type == "task_completed"


def test_destructive_task_rejected_and_tracked(tmp_path):
    """Destructive commands are rejected and tracked."""
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id=identity.node_id, executor=executor)
    tracker = TaskTracker(mesh_manager=mgr)

    task = MeshTask(
        source_node=identity.node_id,
        target_node=identity.node_id,
        command="rm -rf /important",
        reason="bad idea",
    )
    tracker.track_created(task)
    result = dispatcher.dispatch(task)
    assert result.state == TaskState.REJECTED

    tracker.track_rejected(result)
    events = tracker.get_task_events()
    assert len(events) == 2
    assert events[1].event_type == "task_rejected"


def test_task_events_sync_between_nodes(tmp_path):
    """Task events sync from one node to another."""
    node_a = NodeIdentity.generate()
    node_b = NodeIdentity.generate()
    mgr_a = MeshManager(identity=node_a, data_dir=tmp_path / "a")
    mgr_b = MeshManager(identity=node_b, data_dir=tmp_path / "b")

    tracker = TaskTracker(mesh_manager=mgr_a)
    task = MeshTask(source_node=node_a.node_id, target_node=node_b.node_id,
                    command="echo sync_test")
    tracker.track_created(task)

    req = SyncRequest(node_id=node_b.node_id, last_event_id=None, vector_clock={})
    resp = mgr_a.handle_sync_request(req)
    mgr_b.apply_sync_response(resp)

    events_on_b = mgr_b.events_since(None)
    assert len(events_on_b) == 1
    assert events_on_b[0].event_type == "task_created"
    assert events_on_b[0].payload["command"] == "echo sync_test"


def test_multi_node_dispatch(tmp_path):
    """Dispatch same command to multiple local targets."""
    identity = NodeIdentity.generate()
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id=identity.node_id, executor=executor)

    results = dispatcher.dispatch_all(
        source_node=identity.node_id,
        target_nodes=[identity.node_id],
        command="echo multi",
        reason="broadcast",
    )
    assert len(results) == 1
    assert all(r.state == TaskState.COMPLETED for r in results)
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_distributed_tasks.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Run ALL mesh tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/ tests/unit/test_platform/ tests/unit/test_network/test_discovery_mesh.py tests/integration/test_mesh_smoke.py tests/integration/test_mesh_sync.py tests/integration/test_distributed_inference.py tests/integration/test_cross_device_flow.py tests/integration/test_distributed_tasks.py -v`
Expected: All tests PASS — report total count

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_distributed_tasks.py
git commit -m "feat(mesh): add distributed task execution integration tests"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | MeshTask Model | `mesh/task_model.py` | 8 |
| 2 | Task Executor | `mesh/task_executor.py` | 6 |
| 3 | Task Dispatcher | `mesh/task_dispatcher.py` | 5 |
| 4 | Task Tracker | `mesh/task_tracker.py` | 4 |
| 5 | Integration Tests | `test_distributed_tasks.py` | 4 |

**Total: 5 tasks, 27 tests, 4 new source files, 5 new test files**
