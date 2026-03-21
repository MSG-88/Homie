# tests/unit/self_healing/test_guardian.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.guardian import Guardian


class TestGuardian:
    def test_creates_pid_file(self, tmp_path):
        guardian = Guardian(pid_file=tmp_path / "watchdog.pid")
        guardian.write_pid(12345)
        assert (tmp_path / "watchdog.pid").exists()
        assert (tmp_path / "watchdog.pid").read_text().strip() == "12345"

    def test_read_pid(self, tmp_path):
        pid_file = tmp_path / "watchdog.pid"
        pid_file.write_text("54321")
        guardian = Guardian(pid_file=pid_file)
        assert guardian.read_pid() == 54321

    def test_read_pid_returns_none_if_missing(self, tmp_path):
        guardian = Guardian(pid_file=tmp_path / "nope.pid")
        assert guardian.read_pid() is None

    @patch("homie_core.self_healing.guardian.psutil")
    def test_is_alive_true(self, mock_psutil, tmp_path):
        pid_file = tmp_path / "watchdog.pid"
        pid_file.write_text("100")
        guardian = Guardian(pid_file=pid_file)
        mock_psutil.pid_exists.return_value = True
        assert guardian.is_alive() is True

    @patch("homie_core.self_healing.guardian.psutil")
    def test_is_alive_false(self, mock_psutil, tmp_path):
        pid_file = tmp_path / "watchdog.pid"
        pid_file.write_text("100")
        guardian = Guardian(pid_file=pid_file)
        mock_psutil.pid_exists.return_value = False
        assert guardian.is_alive() is False

    def test_cleanup_removes_pid_file(self, tmp_path):
        pid_file = tmp_path / "watchdog.pid"
        pid_file.write_text("100")
        guardian = Guardian(pid_file=pid_file)
        guardian.cleanup()
        assert not pid_file.exists()
