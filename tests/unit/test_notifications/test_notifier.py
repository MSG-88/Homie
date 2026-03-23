"""Tests for the cross-platform Notifier dispatcher."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from homie_core.config import NotificationConfig
from homie_core.notifications.notifier import (
    Notifier,
    _ConsoleBackend,
    _create_platform_backend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_backend(available: bool = True, send_ok: bool = True):
    backend = MagicMock()
    backend.is_available.return_value = available
    backend.send.return_value = send_ok
    return backend


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

class TestPlatformDetection:
    def test_windows_platform_returns_backend(self):
        """On win32 the factory returns *something* (native or console)."""
        backend = _create_platform_backend()
        assert backend is not None
        assert hasattr(backend, "send")
        assert hasattr(backend, "is_available")

    @patch("homie_core.notifications.notifier.sys")
    def test_linux_platform_fallback(self, mock_sys):
        """When notify-send is absent, falls back to console."""
        mock_sys.platform = "linux"
        with patch("shutil.which", return_value=None):
            backend = _create_platform_backend()
        assert type(backend).__name__ == "_ConsoleBackend"

    @patch("homie_core.notifications.notifier.sys")
    def test_darwin_platform_fallback(self, mock_sys):
        """When osascript is absent, falls back to console."""
        mock_sys.platform = "darwin"
        with patch("shutil.which", return_value=None):
            backend = _create_platform_backend()
        assert type(backend).__name__ == "_ConsoleBackend"

    @patch("homie_core.notifications.notifier.sys")
    def test_unknown_platform(self, mock_sys):
        mock_sys.platform = "freebsd"
        backend = _create_platform_backend()
        assert type(backend).__name__ == "_ConsoleBackend"

    @patch("homie_core.notifications.notifier.sys")
    def test_linux_with_notify_send(self, mock_sys):
        """When notify-send exists, returns Linux backend."""
        mock_sys.platform = "linux"
        with patch("shutil.which", return_value="/usr/bin/notify-send"):
            backend = _create_platform_backend()
        assert type(backend).__name__ == "LinuxNotificationBackend"

    @patch("homie_core.notifications.notifier.sys")
    def test_darwin_with_osascript(self, mock_sys):
        """When osascript exists, returns macOS backend."""
        mock_sys.platform = "darwin"
        with patch("shutil.which", return_value="/usr/bin/osascript"):
            backend = _create_platform_backend()
        assert type(backend).__name__ == "MacOSNotificationBackend"


# ---------------------------------------------------------------------------
# Console fallback
# ---------------------------------------------------------------------------

class TestConsoleBackend:
    def test_is_always_available(self):
        assert _ConsoleBackend().is_available() is True

    def test_send_returns_true(self):
        assert _ConsoleBackend().send("title", "msg") is True


# ---------------------------------------------------------------------------
# DND filtering
# ---------------------------------------------------------------------------

class TestDNDFiltering:
    def test_no_dnd_delivers(self):
        cfg = NotificationConfig(dnd_schedule_enabled=False)
        backend = _make_mock_backend()
        notifier = Notifier(config=cfg, backend=backend)

        assert notifier.notify("Hello", "World") is True
        backend.send.assert_called_once_with("Hello", "World")

    def test_dnd_suppresses_normal(self):
        cfg = NotificationConfig(
            dnd_schedule_enabled=True,
            dnd_schedule_start="22:00",
            dnd_schedule_end="07:00",
        )
        backend = _make_mock_backend()
        notifier = Notifier(config=cfg, backend=backend)

        # Simulate calling at 23:00 (inside DND)
        with patch.object(notifier, "_is_in_dnd", return_value=True):
            assert notifier.notify("Hello", "World", priority="normal") is False
        backend.send.assert_not_called()

    def test_dnd_schedule_same_day(self):
        """DND window that does not wrap midnight, e.g. 13:00-14:00."""
        cfg = NotificationConfig(
            dnd_schedule_enabled=True,
            dnd_schedule_start="13:00",
            dnd_schedule_end="14:00",
        )
        notifier = Notifier(config=cfg, backend=_make_mock_backend())
        assert notifier._is_in_dnd("13:30") is True
        assert notifier._is_in_dnd("12:59") is False
        assert notifier._is_in_dnd("14:01") is False

    def test_dnd_schedule_wraps_midnight(self):
        cfg = NotificationConfig(
            dnd_schedule_enabled=True,
            dnd_schedule_start="22:00",
            dnd_schedule_end="07:00",
        )
        notifier = Notifier(config=cfg, backend=_make_mock_backend())
        assert notifier._is_in_dnd("23:00") is True
        assert notifier._is_in_dnd("02:00") is True
        assert notifier._is_in_dnd("06:59") is True
        assert notifier._is_in_dnd("07:01") is False
        assert notifier._is_in_dnd("12:00") is False

    def test_dnd_disabled_ignores_schedule(self):
        cfg = NotificationConfig(
            dnd_schedule_enabled=False,
            dnd_schedule_start="00:00",
            dnd_schedule_end="23:59",
        )
        notifier = Notifier(config=cfg, backend=_make_mock_backend())
        assert notifier._is_in_dnd("12:00") is False


# ---------------------------------------------------------------------------
# Priority bypass
# ---------------------------------------------------------------------------

class TestPriorityBypass:
    def test_critical_bypasses_dnd(self):
        cfg = NotificationConfig(
            dnd_schedule_enabled=True,
            dnd_schedule_start="00:00",
            dnd_schedule_end="23:59",
        )
        backend = _make_mock_backend()
        notifier = Notifier(config=cfg, backend=backend)

        # Force DND active
        with patch.object(notifier, "_is_in_dnd", return_value=True):
            assert notifier.notify("ALERT", "System down", priority="critical") is True
        backend.send.assert_called_once_with("ALERT", "System down")

    def test_high_does_not_bypass_dnd(self):
        cfg = NotificationConfig(
            dnd_schedule_enabled=True,
            dnd_schedule_start="00:00",
            dnd_schedule_end="23:59",
        )
        backend = _make_mock_backend()
        notifier = Notifier(config=cfg, backend=backend)

        with patch.object(notifier, "_is_in_dnd", return_value=True):
            assert notifier.notify("Info", "FYI", priority="high") is False
        backend.send.assert_not_called()


# ---------------------------------------------------------------------------
# Global enable/disable
# ---------------------------------------------------------------------------

class TestGlobalToggle:
    def test_disabled_suppresses_all(self):
        cfg = NotificationConfig(enabled=False)
        backend = _make_mock_backend()
        notifier = Notifier(config=cfg, backend=backend)

        assert notifier.notify("Hello", "World") is False
        assert notifier.notify("ALERT", "Fire", priority="critical") is False
        backend.send.assert_not_called()


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_is_available_delegates(self):
        backend = _make_mock_backend(available=True)
        notifier = Notifier(backend=backend)
        assert notifier.is_available() is True

        backend.is_available.return_value = False
        assert notifier.is_available() is False

    def test_send_failure_returns_false(self):
        backend = _make_mock_backend(send_ok=False)
        notifier = Notifier(backend=backend)
        assert notifier.notify("Oops", "fail") is False
