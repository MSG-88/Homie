"""Tests for settings page generator."""
from __future__ import annotations
import pytest


def test_render_settings_page():
    from homie_app.tray.settings_page import render_settings_page
    html = render_settings_page(session_token="tok-123", api_port=8721)
    assert "<html" in html
    assert "Settings" in html
    assert "tok-123" in html


def test_settings_page_has_sections():
    from homie_app.tray.settings_page import render_settings_page
    html = render_settings_page(session_token="tok", api_port=8721)
    assert "Inference" in html or "inference" in html
    assert "Privacy" in html or "privacy" in html
