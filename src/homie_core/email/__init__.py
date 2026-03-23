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

    def __init__(self, vault, cache_conn: sqlite3.Connection, working_memory=None,
                 model_engine=None):
        self._vault = vault
        self._conn = cache_conn
        self._working_memory = working_memory
        self._model_engine = model_engine
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
                classifier = EmailClassifier(
                    user_email=account_id,
                    model_engine=self._model_engine,
                )

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
        from homie_core.email.classifier import clean_snippet
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
        from homie_core.email.classifier import clean_snippet
        query = "SELECT id, subject, sender, snippet, priority, account_id FROM emails WHERE is_read=0"
        params: list = []
        if account != "all":
            query += " AND account_id=?"
            params.append(account)
        query += " ORDER BY date DESC"
        rows = self._conn.execute(query, params).fetchall()

        grouped: dict[str, list] = {"high": [], "medium": [], "low": []}
        for r in rows:
            entry = {"id": r[0], "subject": r[1], "sender": r[2], "snippet": clean_snippet(r[3] or "")}
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

    def analyze_email(self, message_id: str) -> dict[str, Any]:
        """Deep-analyze a single email using the LLM.

        Returns structured analysis with spam_score, priority, categories,
        intent, action_needed, and summary.
        """
        # Find the message in cache first
        row = self._conn.execute(
            "SELECT subject, sender, recipients, snippet, body, account_id, "
            "spam_score, priority, categories FROM emails WHERE id=?",
            (message_id,),
        ).fetchone()

        if not row:
            return {"error": "Message not found"}

        from homie_core.email.models import EmailMessage
        msg = EmailMessage(
            id=message_id, thread_id="", account_id=row[5], provider="gmail",
            subject=row[0], sender=row[1],
            recipients=json.loads(row[2] or "[]"),
            snippet=row[3], body=row[4],
            spam_score=row[6] or 0.0, priority=row[7] or "medium",
            categories=json.loads(row[8] or "[]"),
        )

        # Fetch full body if not cached
        if not msg.body:
            for provider in self._providers.values():
                try:
                    msg.body = provider.fetch_message_body(message_id)
                    break
                except Exception:
                    continue

        # Try LLM analysis
        classifier = next(iter(self._classifiers.values()), None)
        if classifier:
            llm_result = classifier.llm_classify(msg)
            if llm_result:
                llm_result["heuristic_spam_score"] = msg.spam_score
                llm_result["heuristic_priority"] = msg.priority
                llm_result["subject"] = msg.subject
                llm_result["sender"] = msg.sender
                return llm_result

        # Fallback: return heuristic data
        return {
            "subject": msg.subject,
            "sender": msg.sender,
            "spam_score": msg.spam_score,
            "priority": msg.priority,
            "categories": msg.categories,
            "intent": "LLM unavailable — heuristic classification only",
            "action_needed": msg.priority == "high",
            "summary": msg.snippet or "",
        }

    def triage(self, account: str = "all", max_emails: int = 15) -> dict[str, Any]:
        """Batch-triage unread emails using the LLM.

        Returns structured triage with per-email analysis.
        """
        query = "SELECT id, thread_id, account_id, subject, sender, recipients, snippet, body, spam_score, priority, categories, date FROM emails WHERE is_read=0"
        params: list = []
        if account != "all":
            query += " AND account_id=?"
            params.append(account)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(max_emails)

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return {"status": "No unread emails", "emails": []}

        from homie_core.email.models import EmailMessage
        messages = []
        for r in rows:
            messages.append(EmailMessage(
                id=r[0], thread_id=r[1], account_id=r[2], provider="gmail",
                subject=r[3], sender=r[4],
                recipients=json.loads(r[5] or "[]"),
                snippet=r[6], body=r[7],
                spam_score=r[8] or 0.0, priority=r[9] or "medium",
                categories=json.loads(r[10] or "[]"),
                date=r[11] or 0.0,
            ))

        classifier = next(iter(self._classifiers.values()), None)
        if classifier:
            llm_results = classifier.llm_triage_batch(messages)
            if llm_results:
                # Merge LLM results with message metadata
                results_by_id = {r["id"]: r for r in llm_results if isinstance(r, dict) and "id" in r}
                merged = []
                for msg in messages:
                    llm = results_by_id.get(msg.id, {})
                    merged.append({
                        "id": msg.id,
                        "subject": msg.subject,
                        "sender": msg.sender,
                        "date": msg.date,
                        "heuristic_spam": msg.spam_score,
                        "heuristic_priority": msg.priority,
                        "llm_spam": llm.get("spam_score", msg.spam_score),
                        "llm_priority": llm.get("priority", msg.priority),
                        "categories": llm.get("categories", msg.categories),
                        "intent": llm.get("intent", ""),
                        "action_needed": llm.get("action_needed", False),
                        "summary": llm.get("summary", msg.snippet or ""),
                    })

                # Separate into buckets
                action_needed = [e for e in merged if e["action_needed"]]
                spam = [e for e in merged if e["llm_spam"] > 0.7]
                important = [e for e in merged if not e["action_needed"] and e["llm_spam"] <= 0.7 and e["llm_priority"] in ("high", "medium")]

                return {
                    "status": f"Triaged {len(messages)} emails",
                    "action_needed": action_needed,
                    "important": important,
                    "likely_spam": spam,
                    "all": merged,
                }

        # Fallback: heuristic-only triage with clean snippets
        from homie_core.email.classifier import clean_snippet
        result = []
        for msg in messages:
            result.append({
                "id": msg.id, "subject": msg.subject, "sender": msg.sender,
                "spam_score": round(msg.spam_score, 2),
                "priority": msg.priority,
                "categories": msg.categories,
                "snippet": clean_snippet(msg.snippet or ""),
            })
        return {"status": f"{len(messages)} emails (heuristic only)", "emails": result}

    def get_intelligent_digest(self, days: int = 1) -> str | dict:
        """Generate a natural-language email digest using the LLM.

        Falls back to structured summary if LLM unavailable.
        """
        cutoff = time.time() - (days * 86400)
        rows = self._conn.execute(
            "SELECT id, thread_id, account_id, subject, sender, recipients, "
            "snippet, body, spam_score, priority, categories, date, is_read "
            "FROM emails WHERE date > ? ORDER BY date DESC",
            (cutoff,),
        ).fetchall()

        if not rows:
            return f"No emails in the last {days} day(s)."

        from homie_core.email.models import EmailMessage
        messages = []
        for r in rows:
            messages.append(EmailMessage(
                id=r[0], thread_id=r[1], account_id=r[2], provider="gmail",
                subject=r[3], sender=r[4],
                recipients=json.loads(r[5] or "[]"),
                snippet=r[6], body=r[7],
                spam_score=r[8] or 0.0, priority=r[9] or "medium",
                categories=json.loads(r[10] or "[]"),
                date=r[11] or 0.0, is_read=bool(r[12]),
            ))

        classifier = next(iter(self._classifiers.values()), None)
        if classifier:
            digest = classifier.llm_digest(messages, days=days)
            if digest:
                return digest

        # Fallback: structured summary
        return self.get_summary(days=days)

    def deep_analyze_email(self, message_id: str) -> dict[str, Any]:
        """Deep contextual analysis: deadlines, required actions, impact, draft reply.

        Goes beyond classification — understands what the email means for the user
        and what they should do about it.
        """
        row = self._conn.execute(
            "SELECT subject, sender, recipients, snippet, body, account_id, "
            "spam_score, priority, categories, date FROM emails WHERE id=?",
            (message_id,),
        ).fetchone()

        if not row:
            return {"error": "Message not found"}

        from homie_core.email.models import EmailMessage
        msg = EmailMessage(
            id=message_id, thread_id="", account_id=row[5], provider="gmail",
            subject=row[0], sender=row[1],
            recipients=json.loads(row[2] or "[]"),
            snippet=row[3], body=row[4],
            spam_score=row[6] or 0.0, priority=row[7] or "medium",
            categories=json.loads(row[8] or "[]"),
            date=row[9] or 0.0,
        )

        # Fetch full body if needed
        if not msg.body:
            for provider in self._providers.values():
                try:
                    msg.body = provider.fetch_message_body(message_id)
                    break
                except Exception:
                    continue

        classifier = next(iter(self._classifiers.values()), None)
        if classifier:
            result = classifier.llm_deep_analyze(msg)
            if result:
                result["subject"] = msg.subject
                result["sender"] = msg.sender
                result["message_id"] = message_id
                return result

        return {
            "subject": msg.subject,
            "sender": msg.sender,
            "message_id": message_id,
            "urgency": "no_action",
            "action_detail": "LLM unavailable — cannot analyze",
        }

    # ── Thread operations ────────────────────────────────────────────

    def get_thread_messages(self, thread_id: str, account: str | None = None):
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                return provider.get_thread(thread_id)
            except Exception:
                continue
        return None

    def list_inbox_threads(self, account: str | None = None, start: int = 0,
                           max_results: int = 20) -> list:
        threads = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                threads.extend(provider.get_inbox_threads(start, max_results))
            except Exception:
                pass
        return threads[:max_results]

    def list_starred_threads(self, account: str | None = None, start: int = 0,
                             max_results: int = 20) -> list:
        threads = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                threads.extend(provider.get_starred_threads(start, max_results))
            except Exception:
                pass
        return threads[:max_results]

    def list_spam_threads(self, account: str | None = None, start: int = 0,
                          max_results: int = 20) -> list:
        threads = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                threads.extend(provider.get_spam_threads(start, max_results))
            except Exception:
                pass
        return threads[:max_results]

    def get_unread_counts(self, account: str | None = None) -> dict:
        totals = {"inbox": 0, "spam": 0, "starred": 0, "total": 0}
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                inbox = provider.get_unread_count("INBOX")
                spam = provider.get_unread_count("SPAM")
                starred = provider.get_unread_count("STARRED")
                totals["inbox"] += inbox
                totals["spam"] += spam
                totals["starred"] += starred
                totals["total"] += inbox + spam
            except Exception:
                pass
        return totals

    # ── Draft management ─────────────────────────────────────────────

    def list_drafts(self, account: str | None = None) -> list:
        drafts = []
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                drafts.extend(provider.list_drafts())
            except Exception:
                pass
        return drafts

    def get_draft(self, draft_id: str, account: str | None = None):
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                return provider.get_draft(draft_id)
            except Exception:
                continue
        return None

    def update_draft(self, draft_id: str, to: str, subject: str, body: str,
                     account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.update_draft(draft_id, to, subject, body)
        return ""

    def delete_draft(self, draft_id: str, account: str | None = None) -> None:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                provider.delete_draft(draft_id)
                return
            except Exception:
                continue

    # ── Send / Reply / Forward ───────────────────────────────────────

    def send_email(self, to: str, subject: str, body: str,
                   cc: list[str] | None = None, bcc: list[str] | None = None,
                   attachments: list[str] | None = None,
                   account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.send_email(to, subject, body, cc, bcc, attachments)
        return ""

    def send_draft(self, draft_id: str, account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.send_draft(draft_id)
        return ""

    def reply(self, message_id: str, body: str, send: bool = False,
              account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.reply(message_id, body, send)
        return ""

    def reply_all(self, message_id: str, body: str, send: bool = False,
                  account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.reply_all(message_id, body, send)
        return ""

    def forward(self, message_id: str, to: str, body: str, send: bool = False,
                account: str | None = None) -> str:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            return provider.forward(message_id, to, body, send)
        return ""

    # ── Attachments ──────────────────────────────────────────────────

    def get_attachments(self, message_id: str, account: str | None = None) -> list:
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                return provider.get_attachments(message_id)
            except Exception:
                continue
        return []

    def download_attachment(self, message_id: str, attachment_id: str,
                            save_path: str | None = None,
                            account: str | None = None) -> str:
        from pathlib import Path
        base = Path.home() / ".homie" / "attachments"
        if save_path:
            resolved = Path(save_path).resolve()
            if not str(resolved).startswith(str(base.resolve())):
                return ""
            target = str(resolved)
        else:
            target = str(base / message_id)
        for acct_id, provider in self._providers.items():
            if account and acct_id != account:
                continue
            try:
                return provider.download_attachment(message_id, attachment_id, target)
            except Exception:
                continue
        return ""

    # ── Knowledge & Insights ─────────────────────────────────────────

    def get_contact_insights(self, email_or_name: str):
        try:
            row = self._conn.execute(
                "SELECT email, name, organization, relationship, email_count, "
                "last_contact, topics, entity_id FROM email_contacts "
                "WHERE email=? OR name LIKE ?",
                (email_or_name, f"%{email_or_name}%"),
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        from homie_core.email.models import ContactInsight
        try:
            actions = self._conn.execute(
                "SELECT description FROM email_action_items WHERE assignee=? AND status='pending'",
                (row[0],),
            ).fetchall()
        except Exception:
            actions = []
        return ContactInsight(
            email=row[0], name=row[1] or "", organization=row[2] or "",
            relationship=row[3] or "", email_count=row[4] or 0,
            last_contact=row[5] or 0.0,
            topics=json.loads(row[6] or "[]"),
            pending_actions=[a[0] for a in actions],
        )

    def get_pending_actions(self) -> list:
        from homie_core.email.models import ActionItem
        try:
            rows = self._conn.execute(
                "SELECT id, message_id, thread_id, description, assignee, deadline, "
                "urgency, status, extracted_at FROM email_action_items "
                "WHERE status='pending' ORDER BY deadline ASC NULLS LAST",
            ).fetchall()
        except Exception:
            return []
        return [
            ActionItem(id=r[0], message_id=r[1], thread_id=r[2] or "",
                       description=r[3], assignee=r[4] or "", deadline=r[5],
                       urgency=r[6] or "medium", status=r[7], extracted_at=r[8] or 0.0)
            for r in rows
        ]

    def get_topic_summary(self, topic: str) -> str:
        try:
            row = self._conn.execute(
                "SELECT thread_ids, message_count FROM email_topics WHERE name LIKE ?",
                (f"%{topic}%",),
            ).fetchone()
        except Exception:
            return f"No email threads found for topic '{topic}'"
        if not row:
            return f"No email threads found for topic '{topic}'"
        thread_ids = json.loads(row[0] or "[]")
        return f"Topic '{topic}': {row[1]} messages across {len(thread_ids)} threads"

    def get_email_insights(self, days: int = 1) -> str:
        """Generate intelligence briefing combining digest + pending actions."""
        try:
            digest = self.get_intelligent_digest(days=days)
            if isinstance(digest, str):
                return digest
            return json.dumps(digest)
        except (AttributeError, Exception):
            summary = self.get_summary(days=days)
            actions = self.get_pending_actions()
            result = {**summary, "pending_actions": [
                {"description": a.description, "urgency": a.urgency, "deadline": a.deadline}
                for a in actions[:10]
            ]}
            return json.dumps(result)

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
                    label = provider.create_label(label_name)
                    label_ids[category] = label.id
                except Exception:
                    pass
        return label_ids
