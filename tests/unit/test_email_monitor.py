"""Tests for real-time email monitor."""
import pytest
import time
from unittest.mock import MagicMock
from homie_core.email.realtime_monitor import RealtimeEmailMonitor, EmailAlert


class TestRealtimeEmailMonitor:
    def test_detects_urgent_email(self):
        emails = [{"id": "1", "subject": "URGENT: Server down", "sender": "ops@co.com", "snippet": "Server is down ASAP fix needed"}]
        monitor = RealtimeEmailMonitor(check_fn=lambda: emails, poll_interval=60)
        alerts = monitor.check_now()
        assert len(alerts) == 1
        assert alerts[0].priority == "urgent"
        assert alerts[0].category == "action_required"

    def test_detects_action_required(self):
        emails = [{"id": "2", "subject": "Please review the proposal", "sender": "boss@co.com", "snippet": "Needs your approval by EOD"}]
        monitor = RealtimeEmailMonitor(check_fn=lambda: emails)
        alerts = monitor.check_now()
        assert len(alerts) == 1
        assert alerts[0].priority == "high"

    def test_detects_security_alert(self):
        emails = [{"id": "3", "subject": "Security breach detected", "sender": "security@co.com", "snippet": "Unauthorized access detected"}]
        monitor = RealtimeEmailMonitor(check_fn=lambda: emails)
        alerts = monitor.check_now()
        assert len(alerts) == 1
        assert alerts[0].category == "security"

    def test_ignores_routine_emails(self):
        emails = [{"id": "4", "subject": "Weekly newsletter", "sender": "news@co.com", "snippet": "This week in tech"}]
        monitor = RealtimeEmailMonitor(check_fn=lambda: emails)
        alerts = monitor.check_now()
        assert len(alerts) == 0

    def test_deduplicates_seen_emails(self):
        emails = [{"id": "5", "subject": "URGENT fix", "sender": "a@b.com", "snippet": "ASAP"}]
        monitor = RealtimeEmailMonitor(check_fn=lambda: emails)
        alerts1 = monitor.check_now()
        alerts2 = monitor.check_now()
        assert len(alerts1) == 1
        assert len(alerts2) == 0  # Already seen

    def test_calls_notify_fn(self):
        emails = [{"id": "6", "subject": "URGENT", "sender": "a@b.com", "snippet": "ASAP fix"}]
        notify = MagicMock()
        monitor = RealtimeEmailMonitor(check_fn=lambda: emails, notify_fn=notify)
        monitor.check_now()
        notify.assert_called_once()

    def test_pending_alerts(self):
        emails = [{"id": "7", "subject": "Please review", "sender": "a@b.com", "snippet": "Action required"}]
        monitor = RealtimeEmailMonitor(check_fn=lambda: emails)
        monitor.check_now()
        assert len(monitor.pending_alerts) == 1
        monitor.mark_notified("7")
        assert len(monitor.pending_alerts) == 0

    def test_start_stop(self):
        monitor = RealtimeEmailMonitor(check_fn=lambda: [], poll_interval=1)
        monitor.start()
        assert monitor.is_running
        monitor.stop()
        assert not monitor.is_running

    def test_stats(self):
        emails = [{"id": "8", "subject": "URGENT task", "sender": "a@b.com", "snippet": "ASAP"}]
        monitor = RealtimeEmailMonitor(check_fn=lambda: emails)
        monitor.check_now()
        stats = monitor.get_stats()
        assert stats["total_seen"] == 1
        assert stats["total_alerts"] == 1
