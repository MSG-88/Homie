from __future__ import annotations
from collections import deque
from dataclasses import dataclass


@dataclass
class ActionEvent:
    app: str
    window_title: str
    timestamp: float


CODING_APPS = {
    "code.exe", "code", "pycharm64.exe", "pycharm", "idea64.exe",
    "vim", "nvim", "emacs", "sublime_text",
}
BROWSER_APPS = {
    "chrome.exe", "chrome", "firefox.exe", "firefox", "msedge.exe", "safari",
}
MEETING_APPS = {
    "zoom.exe", "zoom", "teams.exe", "teams", "slack.exe", "slack", "webex",
}
WRITING_KEYWORDS = {"word", "docs.google", "notion", "obsidian", "typora", "write", "draft"}


class ActionDetector:
    """Classifies current user activity from window/app sequence."""

    def __init__(self, window_size: int = 30):
        self._events: deque[ActionEvent] = deque(maxlen=window_size)

    def push(self, app: str, title: str, timestamp: float) -> None:
        self._events.append(ActionEvent(app, title, timestamp))

    def current_activity(self) -> str:
        if not self._events:
            return "idle"
        recent = list(self._events)[-5:]
        apps = {e.app.lower() for e in recent}
        if apps & MEETING_APPS:
            return "meeting"
        if apps & CODING_APPS:
            return "coding"
        titles = " ".join(e.window_title.lower() for e in recent)
        if any(kw in titles for kw in WRITING_KEYWORDS):
            return "writing"
        if apps & BROWSER_APPS:
            return "browsing"
        return "working"

    @property
    def event_count(self) -> int:
        return len(self._events)
