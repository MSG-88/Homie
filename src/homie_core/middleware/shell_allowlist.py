from __future__ import annotations
import re
from homie_core.middleware.base import HomieMiddleware

# Patterns that indicate dangerous shell operations
_DANGEROUS_PATTERNS = [
    r'\$\(',       # command substitution $(
    r'`',          # backtick substitution
    r'\$\{',       # variable expansion ${
    r'[>]{1,2}',   # redirect > or >>
    r'<',          # input redirect
    r'&\s*$',      # background &
    r'\|.*(?:rm|dd|mkfs|format)',  # pipe to dangerous commands
]
_DANGEROUS_RE = re.compile('|'.join(_DANGEROUS_PATTERNS))

DEFAULT_SAFE_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "find", "echo", "pwd", "whoami",
    "date", "git", "python", "pip", "npm", "node", "pytest", "which",
    "where", "dir", "type", "cd", "mkdir", "touch", "wc", "sort", "uniq",
    "env", "printenv", "uname", "hostname",
}


class ShellAllowlistMiddleware(HomieMiddleware):
    name = "shell_allowlist"
    order = 4

    def __init__(self, allowed_commands: set[str] | None = None, mode: str = "allowlist"):
        """mode: 'allowlist' (only allowed commands), 'blocklist' (block dangerous patterns only), 'all' (allow everything)"""
        self._allowed = DEFAULT_SAFE_COMMANDS if allowed_commands is None else allowed_commands
        self._mode = mode

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        if name != "run_command":
            return args
        command = args.get("command", "")
        if self._mode == "all":
            return args
        # Always block dangerous patterns
        if _DANGEROUS_RE.search(command):
            args["_blocked"] = f"Blocked: dangerous shell pattern detected in: {command}"
            return None
        if self._mode == "allowlist":
            # Extract the base command (first word)
            base_cmd = command.strip().split()[0] if command.strip() else ""
            # Handle paths like /usr/bin/git → git
            base_cmd = base_cmd.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if base_cmd not in self._allowed:
                args["_blocked"] = f"Blocked: '{base_cmd}' is not in the allowed commands list"
                return None
        return args
