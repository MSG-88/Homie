from __future__ import annotations

import threading
from typing import Optional


class TrayApp:
    def __init__(
        self,
        on_quit=None,
        on_toggle_voice=None,
        on_open_dashboard=None,
        on_open_briefing=None,
        on_open_chat=None,
    ):
        self._on_quit = on_quit
        self._on_toggle_voice = on_toggle_voice
        self._on_open_dashboard = on_open_dashboard
        self._on_open_briefing = on_open_briefing
        self._on_open_chat = on_open_chat
        self._icon = None
        self._thread: Optional[threading.Thread] = None
        self._voice_enabled = False
        self._unread_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass

    def update_unread_count(self, count: int) -> None:
        self._unread_count = count
        if self._icon:
            suffix = f" — {count} unread" if count > 0 else ""
            self._icon.title = f"Homie AI{suffix}"

    def _run(self) -> None:
        try:
            import pystray
            from PIL import Image

            image = Image.new("RGB", (64, 64), color=(52, 152, 219))
            menu = pystray.Menu(
                pystray.MenuItem("Morning Briefing", self._briefing_clicked),
                pystray.MenuItem("Chat", self._chat_clicked),
                pystray.MenuItem("Dashboard", self._dashboard_clicked),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Toggle Voice", self._voice_clicked),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self._quit_clicked),
            )
            self._icon = pystray.Icon("homie", image, "Homie AI", menu)
            self._icon.run()
        except ImportError:
            pass

    def _chat_clicked(self, icon=None, item=None):
        if self._on_open_chat:
            self._on_open_chat()

    def _briefing_clicked(self, icon=None, item=None):
        if self._on_open_briefing:
            self._on_open_briefing()

    def _dashboard_clicked(self, icon=None, item=None):
        if self._on_open_dashboard:
            self._on_open_dashboard()

    def _voice_clicked(self, icon=None, item=None):
        self._voice_enabled = not self._voice_enabled
        if self._on_toggle_voice:
            self._on_toggle_voice(self._voice_enabled)

    def _quit_clicked(self, icon=None, item=None):
        if self._on_quit:
            self._on_quit()
        self.stop()
