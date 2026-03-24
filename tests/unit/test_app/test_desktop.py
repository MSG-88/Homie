"""Tests for desktop companion launcher."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


def test_generate_session_token():
    from homie_app.desktop import _generate_session_token
    token = _generate_session_token()
    assert isinstance(token, str)
    assert len(token) >= 32


def test_desktop_companion_init():
    from homie_app.desktop import DesktopCompanion
    companion = DesktopCompanion(config_path=None)
    assert companion._session_token is not None
    assert companion._port == 8721
