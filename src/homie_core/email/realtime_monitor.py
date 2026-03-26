"""Real-time email monitor — background polling with notifications.

Polls Gmail at configurable intervals, detects new urgent/important
emails, and surfaces them proactively through the notification system.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmailAlert:
    """An alert for a new important email."""
    email_id: str
    subject: str
    sender: str
    priority: str      # "urgent", "high", "medium"
    category: str      # "action_required", "deadline", "security", etc.
    snippet: str
    timestamp: float
    notified: bool = False


class RealtimeEmailMonitor:
    """Background email monitoring with configurable polling.

    Polls for new emails at regular intervals and generates alerts
    for anything that needs the user's attention.
    """

    def __init__(
        self,
        check_fn: Callable[[], list[dict]],   # Returns list of new email dicts
        notify_fn: Optional[Callable[[EmailAlert], None]] = None,  # Called on new alert
        poll_interval: int = 300,              # Seconds between checks (default 5 min)
        quiet_hours: tuple[int, int] = (22, 7),  # Don't notify between 10PM-7AM
    ):
        self._check_fn = check_fn
        self._notify_fn = notify_fn
        self._poll_interval = poll_interval
        self._quiet_start, self._quiet_end = quiet_hours
        self._seen_ids: set[str] = set()
        self._alerts: list[EmailAlert] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pending_alerts(self) -> list[EmailAlert]:
        """Get unnotified alerts."""
        with self._lock:
            return [a for a in self._alerts if not a.notified]

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="email-monitor")
        self._thread.start()
        logger.info("Email monitor started (interval=%ds)", self._poll_interval)

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Email monitor stopped")

    def check_now(self) -> list[EmailAlert]:
        """Manual check — returns new alerts."""
        return self._check_for_new_emails()

    def mark_notified(self, email_id: str) -> None:
        """Mark an alert as notified."""
        with self._lock:
            for alert in self._alerts:
                if alert.email_id == email_id:
                    alert.notified = True

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                if not self._is_quiet_hours():
                    self._check_for_new_emails()
                else:
                    logger.debug("Quiet hours — skipping email check")
            except Exception as exc:
                logger.error("Email monitor error: %s", exc)

            # Sleep in small increments so stop() is responsive
            for _ in range(self._poll_interval):
                if not self._running:
                    return
                time.sleep(1)

    def _check_for_new_emails(self) -> list[EmailAlert]:
        """Check for new emails and generate alerts."""
        new_alerts = []
        try:
            emails = self._check_fn()
            for email in emails:
                email_id = email.get("id", "")
                if email_id in self._seen_ids:
                    continue
                self._seen_ids.add(email_id)

                alert = self._classify_alert(email)
                if alert:
                    with self._lock:
                        self._alerts.append(alert)
                    new_alerts.append(alert)

                    if self._notify_fn and not self._is_quiet_hours():
                        try:
                            self._notify_fn(alert)
                        except Exception as exc:
                            logger.warning("Notification failed: %s", exc)
        except Exception as exc:
            logger.error("Email check failed: %s", exc)

        return new_alerts

    def _classify_alert(self, email: dict) -> Optional[EmailAlert]:
        """Classify an email and return an alert if it needs attention."""
        subject = email.get("subject", "").lower()
        snippet = email.get("snippet", "").lower()
        sender = email.get("sender", "")
        combined = f"{subject} {snippet}"

        # Urgent keywords
        if any(kw in combined for kw in ["asap", "urgent", "immediately", "critical"]):
            return EmailAlert(
                email_id=email.get("id", ""), subject=email.get("subject", ""),
                sender=sender, priority="urgent", category="action_required",
                snippet=email.get("snippet", "")[:200], timestamp=time.time(),
            )

        # Action required
        if any(kw in combined for kw in ["action required", "please review", "please confirm", "your approval", "respond by"]):
            return EmailAlert(
                email_id=email.get("id", ""), subject=email.get("subject", ""),
                sender=sender, priority="high", category="action_required",
                snippet=email.get("snippet", "")[:200], timestamp=time.time(),
            )

        # Security alerts
        if any(kw in combined for kw in ["security", "breach", "unauthorized", "suspicious", "fraud"]):
            return EmailAlert(
                email_id=email.get("id", ""), subject=email.get("subject", ""),
                sender=sender, priority="high", category="security",
                snippet=email.get("snippet", "")[:200], timestamp=time.time(),
            )

        # Deadline mentions
        if any(kw in combined for kw in ["deadline", "due by", "due date", "expires"]):
            return EmailAlert(
                email_id=email.get("id", ""), subject=email.get("subject", ""),
                sender=sender, priority="medium", category="deadline",
                snippet=email.get("snippet", "")[:200], timestamp=time.time(),
            )

        return None  # Not important enough to alert

    def _is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        from datetime import datetime
        hour = datetime.now().hour
        if self._quiet_start > self._quiet_end:
            return hour >= self._quiet_start or hour < self._quiet_end
        return self._quiet_start <= hour < self._quiet_end

    def get_stats(self) -> dict:
        """Get monitor statistics."""
        with self._lock:
            return {
                "running": self._running,
                "total_seen": len(self._seen_ids),
                "total_alerts": len(self._alerts),
                "pending_alerts": len([a for a in self._alerts if not a.notified]),
            }
