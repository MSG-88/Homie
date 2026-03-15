from __future__ import annotations
from homie_core.middleware.base import HomieMiddleware
from homie_core.backend.protocol import BackendProtocol


class EditConflictMiddleware(HomieMiddleware):
    """Pre-validates edit operations to prevent silent failures."""
    name = "edit_conflict"
    order = 80

    def __init__(self, backend: BackendProtocol):
        self._backend = backend

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        if name not in ("edit_file", "edit"):
            return args
        path = args.get("path", "")
        old = args.get("old", args.get("old_string", ""))
        replace_all = args.get("replace_all", False)
        if not path or not old:
            return args
        try:
            content = self._backend.read(path).content
        except (FileNotFoundError, ValueError):
            args["_conflict"] = f"File not found: {path}"
            return None
        count = content.count(old)
        if count == 0:
            args["_conflict"] = f"String not found in {path} — file may have changed"
            return None
        if count > 1 and not replace_all:
            args["_conflict"] = f"String found {count} times in {path} — not unique. Use replace_all=True or provide more context."
            return None
        return args
