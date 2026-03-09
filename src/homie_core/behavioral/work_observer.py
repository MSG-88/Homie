from __future__ import annotations

from collections import defaultdict
from typing import Any

from homie_core.behavioral.base import BaseObserver


class WorkObserver(BaseObserver):
    IDE_PROCESSES = {"code.exe", "pycharm64.exe", "idea64.exe", "devenv.exe", "webstorm64.exe"}
    BROWSER_PROCESSES = {"chrome.exe", "firefox.exe", "msedge.exe", "brave.exe"}

    def __init__(self):
        super().__init__(name="work")
        self._language_counts: dict[str, int] = defaultdict(int)
        self._tool_counts: dict[str, int] = defaultdict(int)
        self._coding_sessions: int = 0
        self._current_project: str | None = None

    def tick(self) -> dict[str, Any]:
        return {}  # Data comes from context engine, processed in observe()

    def observe(self, active_process: str, window_title: str) -> None:
        self._tool_counts[active_process] += 1

        # Detect language from window title
        lang = self._detect_language(window_title)
        if lang:
            self._language_counts[lang] += 1

        # Detect project from window title
        project = self._detect_project(window_title)
        if project and project != self._current_project:
            self._current_project = project
            self.record({"type": "project_switch", "project": project})

        if active_process.lower() in self.IDE_PROCESSES:
            self.record({"type": "coding", "process": active_process, "title": window_title})

    def get_profile_updates(self) -> dict[str, Any]:
        top_languages = sorted(self._language_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tools = sorted(self._tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "languages": [l[0] for l in top_languages],
            "top_tools": [t[0] for t in top_tools],
            "current_project": self._current_project,
            "coding_sessions": self._coding_sessions,
        }

    def _detect_language(self, title: str) -> str | None:
        ext_map = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".java": "Java",
                   ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".c": "C", ".rb": "Ruby",
                   ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin", ".cs": "C#"}
        for ext, lang in ext_map.items():
            if ext in title.lower():
                return lang
        return None

    def _detect_project(self, title: str) -> str | None:
        if " - " in title:
            parts = title.split(" - ")
            for part in reversed(parts):
                part = part.strip()
                if part and part not in ("Visual Studio Code", "PyCharm", "IntelliJ IDEA"):
                    return part
        return None
