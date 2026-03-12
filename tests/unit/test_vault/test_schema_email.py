"""Tests for email tables in cache.db schema."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from homie_core.vault.schema import create_cache_db


class TestEmailCacheTables:
    def test_emails_table_exists(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='emails'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_emails_insert_and_query(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        conn.execute(
            """INSERT INTO emails (id, thread_id, account_id, provider, subject,
               sender, recipients, snippet, date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("msg1", "t1", "user@gmail.com", "gmail", "Test",
             "alice@x.com", '["user@gmail.com"]', "Hello...", 1710288000.0),
        )
        conn.commit()
        row = conn.execute(
            "SELECT subject FROM emails WHERE id='msg1' AND account_id='user@gmail.com'"
        ).fetchone()
        assert row[0] == "Test"
        conn.close()

    def test_emails_composite_pk(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        for acct in ["a@gmail.com", "b@gmail.com"]:
            conn.execute(
                """INSERT INTO emails (id, thread_id, account_id, provider,
                   subject, sender, recipients, snippet)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("msg1", "t1", acct, "gmail", "Hi", "x@y.com", "[]", "..."),
            )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
        assert count == 2
        conn.close()

    def test_email_sync_state_table(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        conn.execute(
            """INSERT INTO email_sync_state (account_id, provider, history_id)
               VALUES (?, ?, ?)""",
            ("user@gmail.com", "gmail", "12345"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT history_id FROM email_sync_state WHERE account_id='user@gmail.com'"
        ).fetchone()
        assert row[0] == "12345"
        conn.close()

    def test_spam_corrections_table(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        conn.execute(
            """INSERT INTO spam_corrections
               (message_id, account_id, original_score, corrected_action, sender, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("msg1", "user@gmail.com", 0.9, "not_spam", "legit@co.com", 1710288000.0),
        )
        conn.commit()
        row = conn.execute("SELECT corrected_action FROM spam_corrections").fetchone()
        assert row[0] == "not_spam"
        conn.close()

    def test_email_config_table(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        conn.execute(
            """INSERT INTO email_config
               (account_id, check_interval, notify_priority, auto_trash_spam)
               VALUES (?, ?, ?, ?)""",
            ("user@gmail.com", 600, "medium", 0),
        )
        conn.commit()
        row = conn.execute(
            "SELECT check_interval FROM email_config WHERE account_id='user@gmail.com'"
        ).fetchone()
        assert row[0] == 600
        conn.close()

    def test_indexes_created(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_emails%'"
        ).fetchall()
        names = {row[0] for row in indexes}
        assert "idx_emails_account_date" in names
        assert "idx_emails_thread" in names
        assert "idx_emails_priority" in names
        conn.close()

    def test_idempotent_creation(self, tmp_path):
        """Creating cache.db twice should not error (CREATE TABLE IF NOT EXISTS)."""
        db_path = tmp_path / "cache.db"
        conn1 = create_cache_db(db_path)
        conn1.close()
        conn2 = create_cache_db(db_path)
        conn2.execute("SELECT COUNT(*) FROM emails").fetchone()
        conn2.close()
