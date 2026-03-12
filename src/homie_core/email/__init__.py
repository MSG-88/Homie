"""Email integration — provider abstraction, sync, classification, and tools.

EmailService is the main facade used by the daemon and CLI.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from homie_core.email.models import (
    EmailMessage,
    EmailSyncConfig,
    SyncResult,
)


class EmailService:
    """High-level facade for email operations.

    Used by the daemon for sync callbacks and by tools for queries.
    """

    def __init__(self, vault, cache_conn: sqlite3.Connection, working_memory=None):
        self._vault = vault
        self._conn = cache_conn
        self._working_memory = working_memory
        self._providers: dict[str, Any] = {}  # account_id -> GmailProvider
        self._sync_engines: dict[str, Any] = {}  # account_id -> SyncEngine
        self._classifiers: dict[str, Any] = {}  # account_id -> EmailClassifier

    def initialize(self) -> list[str]:
        """Initialize providers for all connected Gmail accounts.

        Returns list of connected account IDs.
        """
        from homie_core.email.gmail_provider import GmailProvider
        from homie_core.email.classifier import EmailClassifier
        from homie_core.email.organizer import EmailOrganizer
        from homie_core.email.sync_engine import SyncEngine

        accounts = []
        credentials = self._vault.list_credentials("gmail")
        client_cred = self._vault.get_credential("gmail", account_id="oauth_client")
        client_id = client_cred.access_token if client_cred else ""
        client_secret = client_cred.refresh_token if client_cred else ""

        for cred in credentials:
            if not cred.active:
                continue
            if cred.account_id == "oauth_client":
                continue
            account_id = cred.account_id
            try:
                provider = GmailProvider(account_id=account_id)
                provider.authenticate(cred, vault=self._vault,
                                      client_id=client_id, client_secret=client_secret)
                classifier = EmailClassifier(user_email=account_id)

                label_ids = self._ensure_homie_labels(provider)

                organizer = EmailOrganizer(provider=provider, label_ids=label_ids)
                engine = SyncEngine(
                    provider=provider,
                    classifier=classifier,
                    cache_conn=self._conn,
                    account_id=account_id,
                    organizer=organizer,
                    vault=self._vault,
                    working_memory=self._working_memory,
                )

                config = self._load_config(account_id)
                self._vault.set_connection_status(
                    "gmail", connected=True, label=account_id,
                    sync_interval=config.check_interval,
                )
                self._providers[account_id] = provider
                self._classifiers[account_id] = classifier
                self._sync_engines[account_id] = engine
                accounts.append(account_id)
            except Exception:
                pass
        return accounts

    def sync_tick(self) -> str:
        """Called by SyncManager on each tick. Syncs all accounts."""
        results = []
        for account_id, engine in self._sync_engines.items():
            config = self._load_config(account_id)
            result = engine.incremental_sync(config=config)
            parts = []
            if result.new_messages:
                parts.append(f"{result.new_messages} new")
            if result.notifications:
                parts.append(f"{len(result.notifications)} notifications")
            if result.errors:
                parts.append(f"{len(result.errors)} errors")
            results.append(f"{account_id}: {', '.join(parts) if parts else 'up to date'}")
        return "; ".join(results) if results else "No accounts"

    def search(self, query: str, account: str = "all", max_results: int = 10) -> list[EmailMessage]:
        """Search emails across accounts."""
        messages = []
        for acct_id, provider in self._providers.items():
            if account != "all" and acct_id != account:
                continue
            try:
                messages.extend(provider.search(query, max_results=max_results))
            except Exception:
                pass
        return messages[:max_results]

    def read_message(self, message_id: str) -> dict[str, Any]:
        """Read full message body."""
        for provider in self._providers.values():
            try:
                body = provider.fetch_message_body(message_id)
                row = self._conn.execute(
                    "SELECT subject, sender, recipients, snippet FROM emails WHERE id=?",
                    (message_id,),
                ).fetchone()
                if row:
                    return {
                        "subject": row[0], "sender": row[1],
                        "recipients": json.loads(row[2] or "[]"),
                        "snippet": row[3], "body": body,
                    }
                return {"body": body}
            except Exception:
                continue
        return {"error": "Message not found"}

    def create_draft(self, to: str, subject: str, body: str,
                     reply_to: str | None = None,
                     cc: list[str] | None = None,
                     bcc: list[str] | None = None,
                     account: str | None = None) -> str:
        """Create a draft via the first available provider."""
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.create_draft(to, subject, body, reply_to, cc, bcc)
        return "no_provider"

    def get_thread(self, thread_id: str) -> list[dict]:
        """Get all messages in a thread from cache."""
        rows = self._conn.execute(
            "SELECT id, subject, sender, snippet, date, priority FROM emails WHERE thread_id=? ORDER BY date",
            (thread_id,),
        ).fetchall()
        return [
            {"id": r[0], "subject": r[1], "sender": r[2], "snippet": r[3], "date": r[4], "priority": r[5]}
            for r in rows
        ]

    def list_labels(self, account: str | None = None) -> list[dict]:
        """List labels from all providers."""
        labels = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                for label in provider.list_labels():
                    labels.append({"id": label.id, "name": label.name, "type": label.type})
            except Exception:
                pass
        return labels

    def get_summary(self, days: int = 1) -> dict:
        """Get email summary for last N days."""
        cutoff = time.time() - (days * 86400)
        rows = self._conn.execute(
            "SELECT priority, is_read, subject, sender FROM emails WHERE date > ? ORDER BY date DESC",
            (cutoff,),
        ).fetchall()
        high_priority = [{"subject": r[2], "sender": r[3]} for r in rows if r[0] == "high"]
        unread = sum(1 for r in rows if not r[1])
        return {"total": len(rows), "unread": unread, "high_priority": high_priority[:10]}

    def get_unread(self, account: str = "all") -> dict:
        """Get unread emails grouped by priority."""
        query = "SELECT id, subject, sender, snippet, priority, account_id FROM emails WHERE is_read=0"
        params: list = []
        if account != "all":
            query += " AND account_id=?"
            params.append(account)
        query += " ORDER BY date DESC"
        rows = self._conn.execute(query, params).fetchall()

        grouped: dict[str, list] = {"high": [], "medium": [], "low": []}
        for r in rows:
            entry = {"id": r[0], "subject": r[1], "sender": r[2], "snippet": r[3]}
            grouped.get(r[4], grouped["medium"]).append(entry)
        return grouped

    def archive_message(self, message_id: str) -> None:
        """Archive a message via provider."""
        for provider in self._providers.values():
            try:
                provider.archive(message_id)
                return
            except Exception:
                continue

    def mark_read(self, message_id: str) -> None:
        """Mark a message as read."""
        for provider in self._providers.values():
            try:
                provider.mark_read(message_id)
                self._conn.execute(
                    "UPDATE emails SET is_read=1 WHERE id=?", (message_id,),
                )
                self._conn.commit()
                return
            except Exception:
                continue

    def _load_config(self, account_id: str) -> EmailSyncConfig:
        """Load sync config from cache.db."""
        row = self._conn.execute(
            "SELECT check_interval, notify_priority, quiet_hours_start, quiet_hours_end, auto_trash_spam "
            "FROM email_config WHERE account_id=?",
            (account_id,),
        ).fetchone()
        if row:
            return EmailSyncConfig(
                account_id=account_id,
                check_interval=row[0],
                notify_priority=row[1],
                quiet_hours_start=row[2],
                quiet_hours_end=row[3],
                auto_trash_spam=bool(row[4]),
            )
        return EmailSyncConfig(account_id=account_id)

    def _ensure_homie_labels(self, provider) -> dict[str, str]:
        """Create Homie/* labels if they don't exist. Returns category->label_id map."""
        from homie_core.email.organizer import HOMIE_LABELS
        existing = {l.name: l.id for l in provider.list_labels()}
        label_ids = {}
        for category, label_name in HOMIE_LABELS.items():
            if label_name in existing:
                label_ids[category] = existing[label_name]
            else:
                try:
                    result = provider._service.users().labels().create(
                        userId="me",
                        body={"name": label_name, "labelListVisibility": "labelShow",
                              "messageListVisibility": "show"},
                    ).execute()
                    label_ids[category] = result["id"]
                except Exception:
                    pass
        return label_ids
