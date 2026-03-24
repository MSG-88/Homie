"""Tests for TrayApp with email integration."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from homie_app.tray.app import TrayApp


def test_tray_app_stores_callbacks():
    on_briefing = MagicMock()
    tray = TrayApp(on_open_briefing=on_briefing)
    assert tray._on_open_briefing is on_briefing


def test_briefing_callback_invoked():
    on_briefing = MagicMock()
    tray = TrayApp(on_open_briefing=on_briefing)
    tray._briefing_clicked()
    on_briefing.assert_called_once()


def test_chat_callback_invoked():
    on_chat = MagicMock()
    tray = TrayApp(on_open_chat=on_chat)
    tray._chat_clicked()
    on_chat.assert_called_once()


def test_tray_accepts_unread_count():
    tray = TrayApp()
    tray.update_unread_count(5)
    assert tray._unread_count == 5
