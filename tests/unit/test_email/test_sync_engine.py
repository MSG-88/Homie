"""Tests for email sync engine — initial + incremental sync."""
from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock, patch

from homie_core.email.sync_engine import SyncEngine
from homie_core.email.models import (
    EmailMessage, EmailSyncConfig, HistoryChange, SyncResult,
)


def _make_msg(msg_id="msg1", **overrides) -> EmailMessage:
    defaults = dict(
        id=msg_id, thread_id="t1", account_id="user@gmail.com",
        provider="gmail", subject="Test", sender="alice@x.com",
        recipients=["user@gmail.com"], snippet="Hello...",
        date=time.time(), priority="medium", spam_score=0.0, categories=[],
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


class TestInitialSync:
    def test_initial_sync_fetches_and_stores(self, tmp_path):
        provider = MagicMock()
        provider.fetch_messages.return_value = [
            _make_msg("m1"), _make_msg("m2"),
        ]
        provider.get_profile.return_value = {"emailAddress": "user@gmail.com", "historyId": "100"}

        classifier = MagicMock()
        classifier.spam_score.return_value = 0.1
        classifier.priority_score.return_value = "medium"
        classifier.detect_categories.return_value = []

        db_path = tmp_path / "cache.db"
        from homie_core.vault.schema import create_cache_db
        conn = create_cache_db(db_path)

        engine = SyncEngine(
            provider=provider,
            classifier=classifier,
            cache_conn=conn,
            account_id="user@gmail.com",
        )
        result = engine.initial_sync()
        assert result.new_messages == 2
        assert result.account_id == "user@gmail.com"

        # Check stored in DB
        count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
        assert count == 2

        # Check sync state
        state = conn.execute(
            "SELECT history_id FROM email_sync_state WHERE account_id='user@gmail.com'"
        ).fetchone()
        assert state[0] == "100"
        conn.close()


class TestIncrementalSync:
    def test_incremental_adds_new_messages(self, tmp_path):
        provider = MagicMock()
        provider.get_history.return_value = (
            [HistoryChange(message_id="m3", change_type="added")],
            "200",
        )
        provider.fetch_messages.return_value = []
        new_msg = _make_msg("m3", subject="New email")
        provider.search.return_value = []

        # Mock the single message fetch (public interface)
        def mock_fetch_message(msg_id):
            if msg_id == "m3":
                return new_msg
            return None
        provider.fetch_message = mock_fetch_message

        classifier = MagicMock()
        classifier.spam_score.return_value = 0.0
        classifier.priority_score.return_value = "high"
        classifier.detect_categories.return_value = []

        db_path = tmp_path / "cache.db"
        from homie_core.vault.schema import create_cache_db
        conn = create_cache_db(db_path)

        # Seed sync state
        conn.execute(
            "INSERT INTO email_sync_state (account_id, provider, history_id) VALUES (?, ?, ?)",
            ("user@gmail.com", "gmail", "100"),
        )
        conn.commit()

        engine = SyncEngine(
            provider=provider,
            classifier=classifier,
            cache_conn=conn,
            account_id="user@gmail.com",
        )
        result = engine.incremental_sync()
        assert result.new_messages == 1
        conn.close()

    def test_incremental_handles_deletions(self, tmp_path):
        provider = MagicMock()
        provider.get_history.return_value = (
            [HistoryChange(message_id="m1", change_type="deleted")],
            "200",
        )

        classifier = MagicMock()

        db_path = tmp_path / "cache.db"
        from homie_core.vault.schema import create_cache_db
        conn = create_cache_db(db_path)

        # Seed existing email + sync state
        conn.execute(
            "INSERT INTO emails (id, thread_id, account_id, provider, subject, sender, recipients, snippet) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("m1", "t1", "user@gmail.com", "gmail", "Old", "x@y.com", "[]", "..."),
        )
        conn.execute(
            "INSERT INTO email_sync_state (account_id, provider, history_id) VALUES (?, ?, ?)",
            ("user@gmail.com", "gmail", "100"),
        )
        conn.commit()

        engine = SyncEngine(
            provider=provider,
            classifier=classifier,
            cache_conn=conn,
            account_id="user@gmail.com",
        )
        result = engine.incremental_sync()
        count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
        assert count == 0  # Deleted
        conn.close()


class TestNotificationDecision:
    def test_high_priority_clean_email_notifies(self):
        config = EmailSyncConfig(account_id="user@gmail.com", notify_priority="high")
        msg = _make_msg(priority="high", spam_score=0.1)
        engine = SyncEngine.__new__(SyncEngine)
        assert engine._should_notify(msg, config) is True

    def test_medium_priority_not_notified_when_high_only(self):
        config = EmailSyncConfig(account_id="user@gmail.com", notify_priority="high")
        msg = _make_msg(priority="medium", spam_score=0.0)
        engine = SyncEngine.__new__(SyncEngine)
        assert engine._should_notify(msg, config) is False

    def test_notify_none_disables_all(self):
        config = EmailSyncConfig(account_id="user@gmail.com", notify_priority="none")
        msg = _make_msg(priority="high", spam_score=0.0)
        engine = SyncEngine.__new__(SyncEngine)
        assert engine._should_notify(msg, config) is False

    def test_spam_never_notifies(self):
        config = EmailSyncConfig(account_id="user@gmail.com", notify_priority="all")
        msg = _make_msg(priority="high", spam_score=0.5)
        engine = SyncEngine.__new__(SyncEngine)
        assert engine._should_notify(msg, config) is False

    def test_quiet_hours_suppresses(self):
        config = EmailSyncConfig(
            account_id="user@gmail.com", notify_priority="high",
            quiet_hours_start=22, quiet_hours_end=7,
        )
        msg = _make_msg(priority="high", spam_score=0.0)
        engine = SyncEngine.__new__(SyncEngine)
        # Mock current hour to 23 (within quiet hours)
        with patch("homie_core.email.sync_engine._current_hour", return_value=23):
            assert engine._should_notify(msg, config) is False
