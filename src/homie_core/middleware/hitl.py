from __future__ import annotations
from homie_core.middleware.base import HomieMiddleware


class HITLMiddleware(HomieMiddleware):
    """Human-in-the-loop approval for dangerous tool calls."""
    name = "hitl"
    order = 3

    def __init__(self, gated_tools: set[str] | None = None):
        """gated_tools: set of tool names requiring approval. Default: {'run_command', 'write_file'}"""
        self._gated = gated_tools or {"run_command", "write_file", "email_send", "email_send_draft", "email_reply", "email_reply_all", "email_forward"}
        self._auto_approved: set[str] = set()  # tools the user has auto-approved

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        if name not in self._gated:
            return args
        if name in self._auto_approved:
            return args
        # Store pending call for the application layer to handle
        args["_hitl_pending"] = {
            "tool": name,
            "args": dict(args),
            "options": ["approve", "reject", "auto_approve"],
        }
        return None  # block execution until approved

    def approve(self, tool_name: str, auto: bool = False) -> None:
        """Called by the application layer after user approves."""
        if auto:
            self._auto_approved.add(tool_name)

    def reject(self) -> None:
        """Called by the application layer after user rejects."""
        pass  # nothing to do — the call was already blocked

    def is_auto_approved(self, tool_name: str) -> bool:
        return tool_name in self._auto_approved
