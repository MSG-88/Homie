from __future__ import annotations

import logging
import subprocess

from homie_core.middleware.base import HomieMiddleware

logger = logging.getLogger(__name__)


class ExternalHooksMiddleware(HomieMiddleware):
    """Execute external commands at middleware lifecycle points."""

    name = "external_hooks"
    order = 95

    def __init__(self, hooks: dict[str, list[str]] | None = None):
        """hooks: mapping of event names to lists of shell commands.

        Events: before_turn, after_turn, on_tool_call, on_tool_result
        """
        self._hooks = hooks or {}

    def _fire(self, event: str, context: dict | None = None) -> None:
        commands = self._hooks.get(event, [])
        for cmd in commands:
            try:
                subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    start_new_session=True,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning(
                    "Hook '%s' failed for event '%s': %s", cmd, event, e
                )

    def before_turn(self, message: str, state: dict) -> str:
        self._fire("before_turn", {"message": message})
        return message

    def after_turn(self, response: str, state: dict) -> str:
        self._fire("after_turn", {"response": response})
        return response

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        self._fire("on_tool_call", {"tool": name, "args": args})
        return args

    def wrap_tool_result(self, name: str, result: str) -> str:
        self._fire("on_tool_result", {"tool": name})
        return result
