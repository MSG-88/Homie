from __future__ import annotations

from pathlib import Path

from homie_core.middleware.base import HomieMiddleware


class MemoryMiddleware(HomieMiddleware):
    """Persistent cross-session memory via AGENTS.md files."""

    name = "memory"
    order = 12

    def __init__(self, memory_paths: list[str | Path] | None = None):
        self._memory_files: list[dict] = []
        for path in (memory_paths or []):
            p = Path(path)
            if p.exists() and p.is_file():
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    self._memory_files.append({"path": str(p), "content": content})
                except OSError:
                    pass

    def modify_prompt(self, system_prompt: str) -> str:
        if not self._memory_files:
            return system_prompt
        lines = ["\n[PERSISTENT MEMORY]"]
        for mf in self._memory_files:
            lines.append(f"--- {mf['path']} ---")
            lines.append(mf["content"])
            lines.append("---")
        return system_prompt + "\n".join(lines) + "\n"

    def reload(self) -> None:
        """Reload memory files from disk (call after agent updates them)."""
        reloaded = []
        for mf in self._memory_files:
            p = Path(mf["path"])
            if p.exists():
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    reloaded.append({"path": mf["path"], "content": content})
                except OSError:
                    reloaded.append(mf)
            else:
                reloaded.append(mf)
        self._memory_files = reloaded
