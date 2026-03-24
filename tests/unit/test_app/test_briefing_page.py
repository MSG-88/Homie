"""Tests for morning briefing HTML page generator."""
from __future__ import annotations
import pytest


def test_render_briefing_page_with_data():
    from homie_app.tray.briefing_page import render_briefing_page
    html = render_briefing_page(
        user_name="Muthu",
        summary={"total": 42, "unread": 5, "high_priority": [
            {"subject": "Contract Review", "sender": "alice@acme.com"},
        ]},
        unread={"high": [{"id": "m1", "subject": "Contract Review", "sender": "alice@acme.com", "snippet": "Please review"}],
                "medium": [], "low": []},
        digest="You have 5 unread emails. 1 needs attention.",
        session_token="tok-123",
        api_port=8721,
    )
    assert "Muthu" in html
    assert "Contract Review" in html
    assert "alice@acme.com" in html
    assert "5" in html
    assert "<html" in html


def test_render_briefing_page_empty():
    from homie_app.tray.briefing_page import render_briefing_page
    html = render_briefing_page(
        user_name="User",
        summary={"total": 0, "unread": 0, "high_priority": []},
        unread={"high": [], "medium": [], "low": []},
        digest="No emails today.",
        session_token="tok",
        api_port=8721,
    )
    assert "0" in html or "inbox zero" in html.lower() or "No emails" in html
    assert "<html" in html
