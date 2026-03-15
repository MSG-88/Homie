from __future__ import annotations
import json
from homie_core.middleware.base import HomieMiddleware


class TodoMiddleware(HomieMiddleware):
    name = "todo"
    order = 15

    def __init__(self):
        self._todos: list[dict] = []

    @property
    def todos(self) -> list[dict]:
        return list(self._todos)

    def modify_tools(self, tools: list[dict]) -> list[dict]:
        todo_tool = {
            "name": "write_todos",
            "description": (
                "Create or update a task list for tracking multi-step work. "
                "Pass a JSON array of {task, status} objects. "
                "Status: pending, in_progress, done."
            ),
        }
        return tools + [todo_tool]

    def modify_prompt(self, system_prompt: str) -> str:
        if not self._todos:
            return system_prompt
        todo_lines = []
        for i, t in enumerate(self._todos, 1):
            status_icon = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}.get(
                t["status"], "[ ]"
            )
            todo_lines.append(f"  {status_icon} {i}. {t['task']}")
        todo_block = "\n[CURRENT TASKS]\n" + "\n".join(todo_lines) + "\n"
        return system_prompt + todo_block

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        if name != "write_todos":
            return args
        todos_data = args.get("todos", "[]")
        if isinstance(todos_data, str):
            try:
                todos_data = json.loads(todos_data)
            except json.JSONDecodeError:
                return args
        if isinstance(todos_data, list):
            self._todos = [
                {"task": t.get("task", ""), "status": t.get("status", "pending")}
                for t in todos_data
                if isinstance(t, dict)
            ]
        return args
