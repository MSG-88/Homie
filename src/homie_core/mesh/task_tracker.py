"""TaskTracker — event-based task lifecycle tracking via MeshManager."""
from __future__ import annotations
from homie_core.mesh.events import HomieEvent
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.task_model import MeshTask


class TaskTracker:
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
        return self._mgr._event_store.events_by_category("task")
