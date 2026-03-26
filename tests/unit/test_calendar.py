"""Tests for Google Calendar integration."""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from homie_core.calendar.gcal_provider import (
    GoogleCalendarProvider, CalendarEvent, get_calendar_context,
)


class TestCalendarEvent:
    def test_duration(self):
        now = datetime.now(timezone.utc)
        event = CalendarEvent(id="1", title="Meeting", start=now, end=now + timedelta(hours=1))
        assert event.duration_minutes() == 60

    def test_is_happening_now(self):
        now = datetime.now(timezone.utc)
        event = CalendarEvent(id="1", title="Now", start=now - timedelta(minutes=10), end=now + timedelta(minutes=50))
        assert event.is_happening_now() is True

    def test_not_happening_now(self):
        now = datetime.now(timezone.utc)
        event = CalendarEvent(id="1", title="Later", start=now + timedelta(hours=2), end=now + timedelta(hours=3))
        assert event.is_happening_now() is False

    def test_starts_within(self):
        now = datetime.now(timezone.utc)
        event = CalendarEvent(id="1", title="Soon", start=now + timedelta(minutes=30), end=now + timedelta(hours=1))
        assert event.starts_within(60) is True
        assert event.starts_within(15) is False

    def test_to_brief(self):
        event = CalendarEvent(
            id="1", title="Team Standup", location="Room 3",
            start=datetime(2026, 3, 26, 14, 0, tzinfo=timezone.utc),
            end=datetime(2026, 3, 26, 14, 30, tzinfo=timezone.utc),
        )
        brief = event.to_brief()
        assert "Team Standup" in brief
        assert "30min" in brief
        assert "Room 3" in brief


class TestGoogleCalendarProvider:
    @patch("homie_core.calendar.gcal_provider.requests")
    def test_connect_success(self, mock_requests):
        mock_requests.request.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value={"items": []})
        )
        provider = GoogleCalendarProvider()
        assert provider.connect("test_token") is True
        assert provider.is_connected is True

    @patch("homie_core.calendar.gcal_provider.requests")
    def test_connect_failure(self, mock_requests):
        mock_requests.request.return_value = MagicMock(status_code=401)
        mock_requests.request.return_value.raise_for_status.side_effect = Exception("Unauthorized")
        provider = GoogleCalendarProvider()
        assert provider.connect("bad_token") is False

    @patch("homie_core.calendar.gcal_provider.requests")
    def test_get_events(self, mock_requests):
        mock_requests.request.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"items": [
                {"id": "1", "summary": "Standup", "start": {"dateTime": "2026-03-26T10:00:00Z"}, "end": {"dateTime": "2026-03-26T10:30:00Z"}},
            ]}),
        )
        mock_requests.request.return_value.raise_for_status = MagicMock()
        provider = GoogleCalendarProvider(access_token="token")
        provider._connected = True
        now = datetime.now(timezone.utc)
        events = provider.get_events(now, now + timedelta(hours=8))
        assert len(events) == 1
        assert events[0].title == "Standup"

    def test_parse_all_day_event(self):
        provider = GoogleCalendarProvider()
        item = {"id": "2", "summary": "Holiday", "start": {"date": "2026-03-26"}, "end": {"date": "2026-03-27"}}
        event = provider._parse_event(item)
        assert event is not None
        assert event.is_all_day is True


class TestGetCalendarContext:
    def test_empty_events(self):
        assert get_calendar_context([]) == ""

    def test_with_events(self):
        now = datetime.now(timezone.utc)
        events = [
            CalendarEvent(id="1", title="Standup", start=now + timedelta(minutes=30), end=now + timedelta(hours=1)),
            CalendarEvent(id="2", title="Lunch", start=now + timedelta(hours=3), end=now + timedelta(hours=4)),
        ]
        ctx = get_calendar_context(events)
        assert "calendar" in ctx.lower()
        assert "Standup" in ctx
        assert "Lunch" in ctx
