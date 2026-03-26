"""Google Calendar provider — read-only access to user's calendar.

Uses Google Calendar API v3 to fetch events. Requires the user to
have connected GCP credentials via /connect gcp or have Application
Default Credentials available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """A calendar event."""
    id: str
    title: str
    start: datetime
    end: datetime
    location: str = ""
    description: str = ""
    attendees: list[str] = field(default_factory=list)
    is_all_day: bool = False
    status: str = "confirmed"   # confirmed, tentative, cancelled
    meeting_link: str = ""

    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    def is_happening_now(self) -> bool:
        now = datetime.now(timezone.utc)
        return self.start <= now <= self.end

    def starts_within(self, minutes: int) -> bool:
        now = datetime.now(timezone.utc)
        return now <= self.start <= now + timedelta(minutes=minutes)

    def to_brief(self) -> str:
        """One-line description for briefings."""
        time_str = self.start.strftime("%I:%M %p")
        dur = self.duration_minutes()
        brief = f"{time_str} — {self.title} ({dur}min)"
        if self.location:
            brief += f" @ {self.location}"
        if self.meeting_link:
            brief += " [online]"
        return brief


class GoogleCalendarProvider:
    """Read-only Google Calendar API provider.

    Uses OAuth2 access token from the vault or Application Default Credentials.
    """

    BASE_URL = "https://www.googleapis.com/calendar/v3"

    def __init__(self, access_token: Optional[str] = None):
        self._token = access_token
        self._connected = False

    def connect(self, access_token: str) -> bool:
        """Set the access token and verify connectivity."""
        self._token = access_token
        try:
            self._call("GET", "/users/me/calendarList", params={"maxResults": 1})
            self._connected = True
            return True
        except Exception as exc:
            logger.warning("Calendar connection failed: %s", exc)
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_today_events(self) -> list[CalendarEvent]:
        """Get all events for today."""
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        return self.get_events(start_of_day, end_of_day)

    def get_upcoming_events(self, hours: int = 4) -> list[CalendarEvent]:
        """Get events in the next N hours."""
        now = datetime.now(timezone.utc)
        return self.get_events(now, now + timedelta(hours=hours))

    def get_events(self, time_min: datetime, time_max: datetime, max_results: int = 20) -> list[CalendarEvent]:
        """Fetch events within a time range."""
        try:
            data = self._call("GET", "/calendars/primary/events", params={
                "timeMin": time_min.isoformat(),
                "timeMax": time_max.isoformat(),
                "maxResults": max_results,
                "singleEvents": "true",
                "orderBy": "startTime",
            })
            events = []
            for item in data.get("items", []):
                event = self._parse_event(item)
                if event:
                    events.append(event)
            return events
        except Exception as exc:
            logger.error("Failed to fetch calendar events: %s", exc)
            return []

    def get_free_busy(self, hours: int = 8) -> dict:
        """Get free/busy status for the next N hours."""
        now = datetime.now(timezone.utc)
        end = now + timedelta(hours=hours)
        try:
            data = self._call("POST", "/freeBusy", json_body={
                "timeMin": now.isoformat(),
                "timeMax": end.isoformat(),
                "items": [{"id": "primary"}],
            })
            busy_periods = data.get("calendars", {}).get("primary", {}).get("busy", [])
            total_busy_mins = sum(
                (datetime.fromisoformat(p["end"].replace("Z", "+00:00")) -
                 datetime.fromisoformat(p["start"].replace("Z", "+00:00"))).total_seconds() / 60
                for p in busy_periods
            )
            return {
                "busy_periods": len(busy_periods),
                "busy_minutes": int(total_busy_mins),
                "free_minutes": int(hours * 60 - total_busy_mins),
            }
        except Exception as exc:
            logger.error("Free/busy check failed: %s", exc)
            return {"busy_periods": 0, "busy_minutes": 0, "free_minutes": hours * 60}

    def _call(self, method: str, path: str, params: dict | None = None, json_body: dict | None = None) -> dict:
        """Make an authenticated API call."""
        url = f"{self.BASE_URL}{path}"
        headers = {"Authorization": f"Bearer {self._token}"}
        resp = requests.request(method, url, headers=headers, params=params, json=json_body, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _parse_event(self, item: dict) -> Optional[CalendarEvent]:
        """Parse a Google Calendar API event into a CalendarEvent."""
        try:
            # Handle all-day vs timed events
            start_data = item.get("start", {})
            end_data = item.get("end", {})

            is_all_day = "date" in start_data and "dateTime" not in start_data

            if is_all_day:
                start = datetime.fromisoformat(start_data["date"]).replace(tzinfo=timezone.utc)
                end = datetime.fromisoformat(end_data["date"]).replace(tzinfo=timezone.utc)
            else:
                start_str = start_data.get("dateTime", "")
                end_str = end_data.get("dateTime", "")
                start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))

            attendees = []
            for att in item.get("attendees", []):
                email = att.get("email", "")
                name = att.get("displayName", email)
                if email:
                    attendees.append(name)

            # Extract meeting link
            meeting_link = ""
            if item.get("hangoutLink"):
                meeting_link = item["hangoutLink"]
            elif item.get("conferenceData", {}).get("entryPoints"):
                for ep in item["conferenceData"]["entryPoints"]:
                    if ep.get("entryPointType") == "video":
                        meeting_link = ep.get("uri", "")
                        break

            return CalendarEvent(
                id=item.get("id", ""),
                title=item.get("summary", "Untitled"),
                start=start, end=end,
                location=item.get("location", ""),
                description=item.get("description", "")[:500],
                attendees=attendees[:20],
                is_all_day=is_all_day,
                status=item.get("status", "confirmed"),
                meeting_link=meeting_link,
            )
        except Exception as exc:
            logger.debug("Failed to parse event: %s", exc)
            return None


def get_calendar_context(events: list[CalendarEvent]) -> str:
    """Format calendar events as context for the system prompt."""
    if not events:
        return ""

    lines = [f"Today's calendar ({len(events)} events):"]

    happening_now = [e for e in events if e.is_happening_now()]
    upcoming = [e for e in events if e.starts_within(60) and not e.is_happening_now()]
    later = [e for e in events if not e.is_happening_now() and not e.starts_within(60)]

    if happening_now:
        lines.append("  NOW:")
        for e in happening_now:
            lines.append(f"    - {e.to_brief()}")

    if upcoming:
        lines.append("  NEXT HOUR:")
        for e in upcoming:
            lines.append(f"    - {e.to_brief()}")

    if later:
        lines.append("  LATER:")
        for e in later[:5]:
            lines.append(f"    - {e.to_brief()}")

    return "\n".join(lines)
