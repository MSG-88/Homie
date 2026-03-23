"""Neural agent communication — bus and task queue."""

from .agent_bus import AgentBus, AgentMessage
from .task_queue import TaskQueue

__all__ = ["AgentBus", "AgentMessage", "TaskQueue"]
