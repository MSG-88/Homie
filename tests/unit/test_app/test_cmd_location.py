"""Tests for /location slash command."""
import pytest
from unittest.mock import MagicMock
from homie_app.console.router import SlashCommandRouter
from homie_app.console.commands.location import register


def test_location_shows_current():
    router = SlashCommandRouter()
    cfg = MagicMock()
    cfg.location = MagicMock(city="Chennai", region="Tamil Nadu", country="IN", timezone="Asia/Kolkata")
    register(router, {})
    result = router.dispatch("/location", **{"config": cfg, "_router": router})
    assert "Chennai" in result


def test_location_not_set():
    router = SlashCommandRouter()
    cfg = MagicMock()
    cfg.location = None
    register(router, {})
    result = router.dispatch("/location", **{"config": cfg, "_router": router})
    assert "not set" in result.lower()


def test_location_set():
    router = SlashCommandRouter()
    cfg = MagicMock()
    cfg.location = None
    register(router, {})
    result = router.dispatch("/location set Chennai", **{"config": cfg, "config_path": None, "_router": router})
    assert "Chennai" in result


def test_location_set_empty():
    router = SlashCommandRouter()
    cfg = MagicMock()
    cfg.location = None
    register(router, {})
    result = router.dispatch("/location set ", **{"config": cfg, "_router": router})
    assert "usage" in result.lower()
