"""Email sync engine — initial and incremental sync with Gmail.

Handles:
- Initial 7-day sync on first connection
- Incremental sync via Gmail historyId deltas
- Notification decisions based on priority + quiet hours
- Cache storage in cache.db
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from homie_core.email.classifier import EmailClassifier
from homie_core.email.models import (
    EmailMessage,
    EmailSyncConfig,
    HistoryChange,
    SyncResult,
    SyncState,
)
from homie_core.email.provider import EmailProvider


def _current_hour() -> int:
    """Get current hour (0-23). Separated for testing."""
    return datetime.now().hour


class SyncEngine:
    """Manages email sync lifecycle for one account."""

    def __init__(
        self,
        provider: EmailProvider,
        classifier: EmailClassifier,
        cache_conn: sqlite3.Connection,
        account_id: str,
        organizer=None,
        vault=None,
        working_memory=None,
    ):
        self._provider = provider
        self._classifier = classifier
        self._conn = cache_conn
        self._account_id = account_id
        self._organizer = organizer
        self._vault = vault
        self._working_memory = working_memory
        self._llm_enabled = classifier.has_llm

    def initial_sync(self) -> SyncResult:
        """Full sync — fetch last 7 days, classify, organize, store."""
        result = SyncResult(account_id=self._account_id)

        try:
            messages = self._provider.fetch_messages(since=0.0, max_results=100)
            profile = self._provider.get_profile()
            history_id = profile.get("historyId", "")

            for msg in messages:
                self._classify_and_organize(msg, result, config=None)

            self._evict_old_emails()

            self._conn.execute(
                """INSERT OR REPLACE INTO email_sync_state
                   (account_id, provider, history_id, last_full_sync, total_synced)
                   VALUES (?, ?, ?, ?, ?)""",
                (self._account_id, "gmail", history_id, time.time(), result.new_messages),
            )
            self._conn.commit()

        except Exception as e:
            result.errors.append(str(e))

        return result

    def incremental_sync(self, config: EmailSyncConfig | None = None) -> SyncResult:
        """Incremental sync via historyId delta."""
        result = SyncResult(account_id=self._account_id)

        row = self._conn.execute(
            "SELECT history_id FROM email_sync_state WHERE account_id=?",
            (self._account_id,),
        ).fetchone()

        if not row or not row[0]:
            return self.initial_sync()

        history_id = row[0]

        try:
            changes, new_history_id = self._provider.get_history(history_id)

            for change in changes:
                if change.change_type == "added":
                    try:
                        msg = self._provider.fetch_message(change.message_id)
                        if msg:
                            self._classify_and_organize(msg, result, config)
                    except Exception:
                        pass

                elif change.change_type == "deleted":
                    self._conn.execute(
                        "DELETE FROM emails WHERE id=? AND account_id=?",
                        (change.message_id, self._account_id),
                    )

                elif change.change_type in ("labelAdded", "labelRemoved"):
                    self._check_spam_correction(change)
                    self._conn.execute(
                        "UPDATE emails SET labels=? WHERE id=? AND account_id=?",
                        (json.dumps(change.labels), change.message_id, self._account_id),
                    )
                    result.updated_messages += 1

            self._conn.execute(
                """UPDATE email_sync_state
                   SET history_id=?, last_incremental_sync=?,
                       total_synced=total_synced+?
                   WHERE account_id=?""",
                (new_history_id, time.time(), result.new_messages, self._account_id),
            )
            self._conn.commit()

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "HttpError 429" in error_str:
                result.errors.append("rate_limited")
            elif "401" in error_str:
                result.errors.append("auth_expired")
            elif "403" in error_str:
                result.errors.append("scope_revoked")
            else:
                result.errors.append(error_str)

        return result

    def _classify_and_organize(self, msg: EmailMessage, result: SyncResult,
                                config: EmailSyncConfig | None) -> None:
        """Classify, organize, store, and optionally notify for one message."""
        msg.spam_score = self._classifier.spam_score(msg)
        msg.priority = self._classifier.priority_score(msg)
        msg.categories = self._classifier.detect_categories(msg)

        # LLM pass for ambiguous emails (heuristic unsure: 0.2-0.85 spam range)
        # or medium priority (heuristic couldn't decide high vs low)
        is_ambiguous = 0.2 <= msg.spam_score <= 0.85 or msg.priority == "medium"
        if self._llm_enabled and is_ambiguous:
            # Fetch full body if not already present
            if not msg.body:
                try:
                    msg.body = self._provider.fetch_message_body(msg.id)
                except Exception:
                    pass
            llm_result = self._classifier.llm_classify(msg)
            if llm_result:
                # Blend heuristic and LLM scores (60% LLM, 40% heuristic)
                msg.spam_score = 0.6 * llm_result["spam_score"] + 0.4 * msg.spam_score
                msg.priority = llm_result["priority"]
                # Merge LLM categories with heuristic categories
                for cat in llm_result.get("categories", []):
                    if cat not in msg.categories:
                        msg.categories.append(cat)
                # Store LLM analysis metadata in body field as suffix
                intent = llm_result.get("intent", "")
                summary = llm_result.get("summary", "")
                action = llm_result.get("action_needed", False)
                if intent or summary:
                    analysis = f"\n\n---\n[Homie Analysis]\nIntent: {intent}\nSummary: {summary}\nAction needed: {'Yes' if action else 'No'}"
                    msg.body = (msg.body or "") + analysis

        if 0.3 <= msg.spam_score <= 0.8 and "review" not in msg.categories:
            msg.categories.append("review")

        if self._organizer:
            try:
                self._organizer.apply_labels(msg)
                if config and self._organizer.should_trash(msg, config):
                    self._provider.trash(msg.id)
                    result.trashed_messages += 1
                    return
                open_count = 999
                if "newsletter" in msg.categories:
                    row = self._conn.execute(
                        "SELECT COUNT(*) FROM ("
                        "  SELECT 1 FROM emails WHERE sender=? AND account_id=? AND is_read=1"
                        "  ORDER BY date DESC LIMIT 3"
                        ")",
                        (msg.sender, self._account_id),
                    ).fetchone()
                    open_count = row[0] if row else 0
                if self._organizer.should_archive(msg, sender_open_count=open_count):
                    self._provider.archive(msg.id)
            except Exception:
                pass

            if "bill" in msg.categories and self._vault:
                try:
                    fin_data = self._organizer.extract_financial(msg)
                    if fin_data:
                        self._vault.store_financial(**fin_data)
                except Exception:
                    pass

        self._store_message(msg)
        result.new_messages += 1

        if config and self._should_notify(msg, config):
            result.notifications.append(msg)
            if self._working_memory:
                self._working_memory.update("email_alert", {
                    "subject": msg.subject,
                    "sender": msg.sender,
                    "priority": msg.priority,
                    "snippet": msg.snippet,
                })

    def _evict_old_emails(self) -> None:
        """Remove emails older than 90 days from cache."""
        cutoff = time.time() - (90 * 86400)
        self._conn.execute(
            "DELETE FROM emails WHERE account_id=? AND date < ?",
            (self._account_id, cutoff),
        )

    def _check_spam_correction(self, change: HistoryChange) -> None:
        """Detect spam corrections from label changes."""
        if "TRASH" not in change.labels:
            return
        row = self._conn.execute(
            "SELECT spam_score, sender FROM emails WHERE id=? AND account_id=?",
            (change.message_id, self._account_id),
        ).fetchone()
        if not row:
            return
        original_score, sender = row
        action = "is_spam" if change.change_type == "labelAdded" else "not_spam"
        self._conn.execute(
            """INSERT INTO spam_corrections
               (message_id, account_id, original_score, corrected_action, sender, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (change.message_id, self._account_id, original_score, action, sender, time.time()),
        )

    def _store_message(self, msg: EmailMessage) -> None:
        """Store/update a message in cache.db."""
        self._conn.execute(
            """INSERT OR REPLACE INTO emails
               (id, thread_id, account_id, provider, subject, sender,
                recipients, snippet, body, labels, date, is_read, is_starred,
                has_attachments, attachment_names, priority, spam_score,
                categories, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg.id, msg.thread_id, msg.account_id, msg.provider,
                msg.subject, msg.sender,
                json.dumps(msg.recipients), msg.snippet, msg.body,
                json.dumps(msg.labels), msg.date,
                int(msg.is_read), int(msg.is_starred),
                int(msg.has_attachments), json.dumps(msg.attachment_names),
                msg.priority, msg.spam_score,
                json.dumps(msg.categories), time.time(),
            ),
        )

    def _should_notify(self, msg: EmailMessage, config: EmailSyncConfig) -> bool:
        """Decide whether to send a notification for this message."""
        if config.notify_priority == "none":
            return False

        if msg.spam_score >= 0.3:
            return False

        priority_order = {"high": 3, "medium": 2, "low": 1, "all": 0}
        msg_level = priority_order.get(msg.priority, 1)
        threshold = priority_order.get(config.notify_priority, 3)
        if msg_level < threshold:
            return False

        if config.quiet_hours_start is not None and config.quiet_hours_end is not None:
            hour = _current_hour()
            if config.quiet_hours_start > config.quiet_hours_end:
                if hour >= config.quiet_hours_start or hour < config.quiet_hours_end:
                    return False
            else:
                if config.quiet_hours_start <= hour < config.quiet_hours_end:
                    return False

        return True
