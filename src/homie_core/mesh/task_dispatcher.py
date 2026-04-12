from __future__ import annotations
import logging
from typing import Callable, Optional
from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.mesh.task_executor import MeshTaskExecutor

logger = logging.getLogger(__name__)


class TaskDispatcher:
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
        if task.target_node == self._local_node_id:
            logger.debug("Dispatching task %s locally", task.task_id)
            self._executor.execute(task)
        elif self._remote_handler:
            logger.debug("Dispatching task %s to remote node %s via handler", task.task_id, task.target_node)
            task = self._remote_handler(task)
        else:
            logger.warning("No route to remote node %s — marking task failed", task.target_node)
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
        return [
            self.dispatch(MeshTask(source_node=source_node, target_node=t, command=command, reason=reason))
            for t in target_nodes
        ]

    def task_history(self) -> list[MeshTask]:
        return list(self._history)
