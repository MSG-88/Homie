from unittest.mock import patch, MagicMock
from homie_core.screen_reader.window_tracker import WindowTracker, WindowInfo


class TestWindowTracker:
    def test_window_info_dataclass(self):
        info = WindowInfo(title="VS Code", process_name="code.exe", pid=1234)
        assert info.title == "VS Code"
        assert info.process_name == "code.exe"

    @patch("homie_core.screen_reader.window_tracker.win32gui")
    @patch("homie_core.screen_reader.window_tracker.psutil")
    def test_get_active_window(self, mock_psutil, mock_win32gui):
        mock_win32gui.GetForegroundWindow.return_value = 12345
        mock_win32gui.GetWindowText.return_value = "test.py - VS Code"
        mock_win32gui.GetWindowThreadProcessId.return_value = (0, 5678)
        mock_proc = MagicMock()
        mock_proc.name.return_value = "code.exe"
        mock_psutil.Process.return_value = mock_proc

        tracker = WindowTracker()
        info = tracker.get_active_window()
        assert info.title == "test.py - VS Code"
        assert info.process_name == "code.exe"
        assert info.pid == 5678

    def test_has_changed_detects_switch(self):
        tracker = WindowTracker()
        old = WindowInfo(title="Chrome", process_name="chrome.exe", pid=1)
        new = WindowInfo(title="VS Code", process_name="code.exe", pid=2)
        assert tracker.has_changed(old, new) is True

    def test_has_changed_same_window(self):
        tracker = WindowTracker()
        w = WindowInfo(title="Chrome", process_name="chrome.exe", pid=1)
        assert tracker.has_changed(w, w) is False

    def test_blocklist_match(self):
        tracker = WindowTracker(blocklist=["*password*", "*banking*"])
        assert tracker.is_blocked("1Password - Vault") is True
        assert tracker.is_blocked("VS Code - main.py") is False
        assert tracker.is_blocked("Chase Banking - Chrome") is True
