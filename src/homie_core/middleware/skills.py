from __future__ import annotations

import re
from pathlib import Path

from homie_core.middleware.base import HomieMiddleware


class SkillsMiddleware(HomieMiddleware):
    """Progressive disclosure of skills via SKILL.md files."""

    name = "skills"
    order = 20

    def __init__(self, skill_paths: list[str | Path] | None = None):
        self._skills: list[dict] = []
        for path in (skill_paths or []):
            self._load_skill(Path(path))

    def _load_skill(self, path: Path) -> None:
        if not path.exists() or path.stat().st_size > 10 * 1024 * 1024:  # 10 MB cap
            return
        content = path.read_text(encoding="utf-8", errors="replace")
        # Parse YAML frontmatter for name and description
        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return
        frontmatter = match.group(1)
        name = ""
        description = ""
        for line in frontmatter.splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip("\"'")
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip("\"'")
        if name:
            self._skills.append(
                {
                    "name": name,
                    "description": description,
                    "path": str(path),
                }
            )

    def modify_prompt(self, system_prompt: str) -> str:
        if not self._skills:
            return system_prompt
        lines = ["\n[AVAILABLE SKILLS]"]
        for s in self._skills:
            lines.append(
                f"- {s['name']}: {s['description']} (read {s['path']} for full instructions)"
            )
        return system_prompt + "\n".join(lines) + "\n"
