# Email Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Homie to connect to Gmail, read/analyze emails, detect spam, organize inbox, create draft replies, and proactively notify the user about important emails.

**Architecture:** Abstract `EmailProvider` interface with Gmail as first implementation. OAuth 2.0 for auth, incremental sync via Gmail historyId, weighted heuristic classifier for spam/priority, 9 AI tools registered with ToolRegistry, ProactiveEngine integration for notifications.

**Tech Stack:** google-api-python-client, google-auth-oauthlib, google-auth-httplib2, SQLite (cache.db), existing SecureVault for credential storage.

**Spec:** `docs/superpowers/specs/2026-03-12-email-integration-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/homie_core/email/__init__.py` | Re-exports `EmailService`, `GmailProvider` |
| `src/homie_core/email/models.py` | Data classes: `EmailMessage`, `EmailThread`, `HistoryChange`, `SyncState`, `Label`, `EmailSyncConfig`, `SyncResult` |
| `src/homie_core/email/provider.py` | Abstract `EmailProvider` interface (ABC) |
| `src/homie_core/email/oauth.py` | OAuth 2.0 flow: local redirect server + manual fallback |
| `src/homie_core/email/gmail_provider.py` | Gmail API implementation of `EmailProvider` |
| `src/homie_core/email/sync_engine.py` | `SyncEngine` — initial + incremental sync, notification decisions |
| `src/homie_core/email/classifier.py` | Spam scoring + priority scoring + category detection |
| `src/homie_core/email/organizer.py` | Auto-labeling, archive rules, financial extraction |
| `src/homie_core/email/tools.py` | 9 AI tool wrappers for `ToolRegistry` |
| `tests/unit/test_email/test_models.py` | Tests for data models |
| `tests/unit/test_email/test_classifier.py` | Tests for classifier |
| `tests/unit/test_email/test_organizer.py` | Tests for organizer |
| `tests/unit/test_email/test_sync_engine.py` | Tests for sync engine |
| `tests/unit/test_email/test_oauth.py` | Tests for OAuth flow |
| `tests/unit/test_email/test_gmail_provider.py` | Tests for Gmail provider |
| `tests/unit/test_email/test_tools.py` | Tests for AI tools |
| `tests/unit/test_email/__init__.py` | Test package init |

### Modified Files
| File | Change |
|------|--------|
| `src/homie_core/vault/schema.py:65-95` | Extend `_CACHE_DDL` with 4 email tables + indexes |
| `src/homie_app/cli.py` | Wire `homie connect gmail`, add `homie email` subcommands |
| `src/homie_app/daemon.py` | Initialize EmailService, register sync callback |
| `pyproject.toml:82` | Add `email` optional dependency group, update `all` |

---

## Chunk 1: Data Models + Database Schema (Tasks 1-3)

### Task 1: Data Models (`models.py`)

**Files:**
- Create: `src/homie_core/email/__init__.py`
- Create: `src/homie_core/email/models.py`
- Create: `tests/unit/test_email/__init__.py`
- Create: `tests/unit/test_email/test_models.py`

- [ ] **Step 1: Create package init and empty models file**

Create `src/homie_core/email/__init__.py`:
```python
"""Email integration — provider abstraction, sync, classification, and tools."""
```

Create `src/homie_core/email/models.py`:
```python
"""Email data models."""
from __future__ import annotations

from dataclasses import dataclass, field
```

- [ ] **Step 2: Write failing tests for EmailMessage**

Create `tests/unit/test_email/__init__.py` (empty).

Create `tests/unit/test_email/test_models.py`:
```python
"""Tests for email data models."""
from __future__ import annotations

import json

from homie_core.email.models import (
    EmailMessage,
    EmailThread,
    HistoryChange,
    Label,
    SyncState,
    EmailSyncConfig,
    SyncResult,
)


class TestEmailMessage:
    def test_create_minimal(self):
        msg = EmailMessage(
            id="msg1",
            thread_id="t1",
            account_id="user@gmail.com",
            provider="gmail",
            subject="Hello",
            sender="Alice <alice@example.com>",
            recipients=["user@gmail.com"],
            snippet="Hey there...",
        )
        assert msg.id == "msg1"
        assert msg.provider == "gmail"
        assert msg.body is None
        assert msg.priority == "medium"
        assert msg.spam_score == 0.0
        assert msg.is_read is True
        assert msg.categories == []
        assert msg.labels == []

    def test_create_full(self):
        msg = EmailMessage(
            id="msg2",
            thread_id="t2",
            account_id="user@gmail.com",
            provider="gmail",
            subject="Invoice #123",
            sender="billing@util.com",
            recipients=["user@gmail.com", "boss@work.com"],
            snippet="Your invoice is ready",
            body="Full body text here",
            labels=["INBOX", "IMPORTANT"],
            date=1710288000.0,
            is_read=False,
            is_starred=True,
            has_attachments=True,
            attachment_names=["invoice.pdf"],
            priority="high",
            spam_score=0.1,
            categories=["bill"],
        )
        assert msg.is_starred is True
        assert msg.has_attachments is True
        assert msg.attachment_names == ["invoice.pdf"]
        assert msg.priority == "high"

    def test_to_dict(self):
        msg = EmailMessage(
            id="msg1", thread_id="t1", account_id="a@b.com",
            provider="gmail", subject="Hi", sender="x@y.com",
            recipients=["a@b.com"], snippet="...",
        )
        d = msg.to_dict()
        assert d["id"] == "msg1"
        assert d["provider"] == "gmail"
        assert isinstance(d["recipients"], list)

    def test_from_dict(self):
        data = {
            "id": "msg1", "thread_id": "t1", "account_id": "a@b.com",
            "provider": "gmail", "subject": "Hi", "sender": "x@y.com",
            "recipients": ["a@b.com"], "snippet": "...",
        }
        msg = EmailMessage.from_dict(data)
        assert msg.id == "msg1"
        assert msg.recipients == ["a@b.com"]


class TestEmailThread:
    def test_create(self):
        thread = EmailThread(
            id="t1", account_id="user@gmail.com", subject="Discussion",
            participants=["alice@x.com", "bob@y.com"],
            message_count=5, last_message_date=1710288000.0,
            snippet="Latest reply...",
        )
        assert thread.message_count == 5
        assert thread.labels == []


class TestHistoryChange:
    def test_create(self):
        change = HistoryChange(
            message_id="msg1", change_type="added",
        )
        assert change.labels == []

    def test_with_labels(self):
        change = HistoryChange(
            message_id="msg1", change_type="labelAdded",
            labels=["INBOX", "IMPORTANT"],
        )
        assert len(change.labels) == 2


class TestSyncState:
    def test_defaults(self):
        state = SyncState(account_id="user@gmail.com", provider="gmail")
        assert state.history_id is None
        assert state.last_full_sync == 0.0
        assert state.total_synced == 0


class TestLabel:
    def test_defaults(self):
        label = Label(id="Label_1", name="Homie/Bills")
        assert label.type == "user"


class TestEmailSyncConfig:
    def test_defaults(self):
        config = EmailSyncConfig(account_id="user@gmail.com")
        assert config.check_interval == 300
        assert config.notify_priority == "high"
        assert config.quiet_hours_start is None
        assert config.auto_trash_spam is True

    def test_custom(self):
        config = EmailSyncConfig(
            account_id="user@gmail.com",
            check_interval=600,
            notify_priority="medium",
            quiet_hours_start=22,
            quiet_hours_end=7,
            auto_trash_spam=False,
        )
        assert config.check_interval == 600
        assert config.quiet_hours_start == 22


class TestSyncResult:
    def test_defaults(self):
        result = SyncResult(account_id="user@gmail.com")
        assert result.new_messages == 0
        assert result.notifications == []
        assert result.errors == []

    def test_with_data(self):
        msg = EmailMessage(
            id="m1", thread_id="t1", account_id="user@gmail.com",
            provider="gmail", subject="Urgent", sender="boss@work.com",
            recipients=["user@gmail.com"], snippet="Need this ASAP",
        )
        result = SyncResult(
            account_id="user@gmail.com",
            new_messages=3, notifications=[msg],
        )
        assert result.new_messages == 3
        assert len(result.notifications) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_models.py -v`
Expected: ImportError — classes not yet defined.

- [ ] **Step 4: Implement all data models**

Write `src/homie_core/email/models.py`:
```python
"""Email data models.

All models are plain dataclasses with no external dependencies.
Serialization uses to_dict/from_dict for database storage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmailMessage:
    """A single email message with Homie-assigned metadata."""
    id: str
    thread_id: str
    account_id: str
    provider: str
    subject: str
    sender: str
    recipients: list[str]
    snippet: str
    body: str | None = None
    labels: list[str] = field(default_factory=list)
    date: float = 0.0
    is_read: bool = True
    is_starred: bool = False
    has_attachments: bool = False
    attachment_names: list[str] = field(default_factory=list)
    priority: str = "medium"
    spam_score: float = 0.0
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "account_id": self.account_id,
            "provider": self.provider,
            "subject": self.subject,
            "sender": self.sender,
            "recipients": self.recipients,
            "snippet": self.snippet,
            "body": self.body,
            "labels": self.labels,
            "date": self.date,
            "is_read": self.is_read,
            "is_starred": self.is_starred,
            "has_attachments": self.has_attachments,
            "attachment_names": self.attachment_names,
            "priority": self.priority,
            "spam_score": self.spam_score,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmailMessage:
        return cls(
            id=data["id"],
            thread_id=data["thread_id"],
            account_id=data["account_id"],
            provider=data["provider"],
            subject=data["subject"],
            sender=data["sender"],
            recipients=data.get("recipients", []),
            snippet=data.get("snippet", ""),
            body=data.get("body"),
            labels=data.get("labels", []),
            date=data.get("date", 0.0),
            is_read=data.get("is_read", True),
            is_starred=data.get("is_starred", False),
            has_attachments=data.get("has_attachments", False),
            attachment_names=data.get("attachment_names", []),
            priority=data.get("priority", "medium"),
            spam_score=data.get("spam_score", 0.0),
            categories=data.get("categories", []),
        )


@dataclass
class EmailThread:
    """A conversation thread grouping multiple messages."""
    id: str
    account_id: str
    subject: str
    participants: list[str]
    message_count: int
    last_message_date: float
    snippet: str
    labels: list[str] = field(default_factory=list)


@dataclass
class HistoryChange:
    """A single change from Gmail's history API."""
    message_id: str
    change_type: str  # "added", "deleted", "labelAdded", "labelRemoved"
    labels: list[str] = field(default_factory=list)


@dataclass
class SyncState:
    """Tracks sync progress for one account."""
    account_id: str
    provider: str
    history_id: str | None = None
    last_full_sync: float = 0.0
    last_incremental_sync: float = 0.0
    total_synced: int = 0


@dataclass
class Label:
    """An email label/folder."""
    id: str
    name: str
    type: str = "user"  # "system" or "user"


@dataclass
class EmailSyncConfig:
    """Per-account sync and notification settings."""
    account_id: str
    check_interval: int = 300
    notify_priority: str = "high"
    quiet_hours_start: int | None = None
    quiet_hours_end: int | None = None
    auto_trash_spam: bool = True


@dataclass
class SyncResult:
    """Result of a sync operation."""
    account_id: str
    new_messages: int = 0
    updated_messages: int = 0
    trashed_messages: int = 0
    notifications: list[EmailMessage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_models.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/email/__init__.py src/homie_core/email/models.py tests/unit/test_email/__init__.py tests/unit/test_email/test_models.py
git commit -m "feat(email): add data models — EmailMessage, EmailThread, SyncState, etc."
```

---

### Task 2: Database Schema Extension

**Files:**
- Modify: `src/homie_core/vault/schema.py:65-95`
- Test: `tests/unit/test_vault/test_schema_email.py`

- [ ] **Step 1: Write failing test for email tables**

Create `tests/unit/test_vault/test_schema_email.py`:
```python
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
        # Same message ID, different accounts — should work
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
        # Should not raise
        conn2.execute("SELECT COUNT(*) FROM emails").fetchone()
        conn2.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_vault/test_schema_email.py -v`
Expected: FAIL — `emails` table does not exist.

- [ ] **Step 3: Extend `_CACHE_DDL` in schema.py**

In `src/homie_core/vault/schema.py`, append to the `_CACHE_DDL` string (before the closing `"""`):

```python
# After the connection_status table (line 94), add:

CREATE TABLE IF NOT EXISTS emails (
    id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    subject TEXT,
    sender TEXT,
    recipients TEXT,
    snippet TEXT,
    body TEXT,
    labels TEXT,
    date REAL,
    is_read INTEGER DEFAULT 1,
    is_starred INTEGER DEFAULT 0,
    has_attachments INTEGER DEFAULT 0,
    attachment_names TEXT,
    priority TEXT DEFAULT 'medium',
    spam_score REAL DEFAULT 0.0,
    categories TEXT,
    fetched_at REAL,
    PRIMARY KEY (id, account_id)
);

CREATE TABLE IF NOT EXISTS email_sync_state (
    account_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    history_id TEXT,
    last_full_sync REAL,
    last_incremental_sync REAL,
    total_synced INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS spam_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    original_score REAL,
    corrected_action TEXT,
    sender TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS email_config (
    account_id TEXT PRIMARY KEY,
    check_interval INTEGER DEFAULT 300,
    notify_priority TEXT DEFAULT 'high',
    quiet_hours_start INTEGER,
    quiet_hours_end INTEGER,
    auto_trash_spam INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_emails_account_date ON emails(account_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_emails_thread ON emails(thread_id);
CREATE INDEX IF NOT EXISTS idx_emails_priority ON emails(priority, date DESC);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_vault/test_schema_email.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Run full vault test suite to check no regressions**

Run: `pytest tests/unit/test_vault/ -v`
Expected: All existing vault tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/vault/schema.py tests/unit/test_vault/test_schema_email.py
git commit -m "feat(email): extend cache.db schema with email tables and indexes"
```

---

### Task 3: Provider Abstraction (`provider.py`)

**Files:**
- Create: `src/homie_core/email/provider.py`

- [ ] **Step 1: Write the abstract provider interface**

Create `src/homie_core/email/provider.py`:
```python
"""Abstract email provider interface.

Gmail implements this directly via google-api-python-client.
Future providers (Outlook, IMAP) implement the same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from homie_core.email.models import (
    EmailMessage,
    HistoryChange,
    Label,
)


class EmailProvider(ABC):
    """Abstract email provider — one implementation per service."""

    @abstractmethod
    def authenticate(self, credential) -> None:
        """Authenticate with stored Credential (from vault). Refreshes token if expired.

        Args:
            credential: A vault Credential dataclass instance with attributes:
                access_token, refresh_token, expires_at, scopes, account_id, etc.
        """

    @abstractmethod
    def fetch_messages(self, since: float, max_results: int = 100) -> list[EmailMessage]:
        """Fetch messages newer than `since` timestamp."""

    @abstractmethod
    def fetch_message_body(self, message_id: str) -> str:
        """Fetch full body text of a specific message."""

    @abstractmethod
    def get_history(self, start_history_id: str) -> tuple[list[HistoryChange], str]:
        """Get changes since history_id. Returns (changes, new_history_id)."""

    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> list[EmailMessage]:
        """Search messages using provider-native query syntax."""

    @abstractmethod
    def apply_label(self, message_id: str, label_id: str) -> None:
        """Apply a label to a message."""

    @abstractmethod
    def remove_label(self, message_id: str, label_id: str) -> None:
        """Remove a label from a message."""

    @abstractmethod
    def trash(self, message_id: str) -> None:
        """Move a message to trash."""

    @abstractmethod
    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        reply_to: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> str:
        """Create a draft email. Returns draft ID. Never sends."""

    @abstractmethod
    def list_labels(self) -> list[Label]:
        """List all labels/folders for the account."""

    @abstractmethod
    def get_profile(self) -> dict:
        """Get account profile (email address, display name)."""

    @abstractmethod
    def mark_read(self, message_id: str) -> None:
        """Mark a message as read."""

    @abstractmethod
    def archive(self, message_id: str) -> None:
        """Archive a message (remove from inbox)."""
```

- [ ] **Step 2: Commit**

```bash
git add src/homie_core/email/provider.py
git commit -m "feat(email): add abstract EmailProvider interface"
```
## Chunk 2: Classifier + Organizer (Tasks 4-5)

### Task 4: Spam & Priority Classifier (`classifier.py`)

**Files:**
- Create: `src/homie_core/email/classifier.py`
- Create: `tests/unit/test_email/test_classifier.py`

- [ ] **Step 1: Write failing tests for spam scoring**

Create `tests/unit/test_email/test_classifier.py`:
```python
"""Tests for email spam and priority classifier."""
from __future__ import annotations

from homie_core.email.classifier import EmailClassifier
from homie_core.email.models import EmailMessage


def _make_msg(**overrides) -> EmailMessage:
    """Helper to create test EmailMessage with defaults."""
    defaults = dict(
        id="msg1", thread_id="t1", account_id="user@work.com",
        provider="gmail", subject="Hello", sender="alice@example.com",
        recipients=["user@work.com"], snippet="Hey there...",
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


class TestSpamScoring:
    def test_clean_email_from_known_contact(self):
        classifier = EmailClassifier(
            user_email="user@work.com",
            reply_history={"alice@example.com"},
        )
        msg = _make_msg(sender="alice@example.com")
        score = classifier.spam_score(msg)
        assert score < 0.3, f"Known contact should score low, got {score}"

    def test_bulk_sender_scores_higher(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            sender="noreply@marketing.com",
            subject="AMAZING DEAL!!!",
            snippet="Unsubscribe from this list",
        )
        # Simulate headers via labels (bulk indicator)
        score = classifier.spam_score(msg, headers={"Precedence": "bulk", "List-Unsubscribe": "yes"})
        assert score > 0.3, f"Bulk sender should score higher, got {score}"

    def test_same_domain_reduces_score(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(sender="boss@work.com")
        score = classifier.spam_score(msg)
        assert score < 0.3, f"Same domain should score low, got {score}"

    def test_score_clamped_to_0_1(self):
        classifier = EmailClassifier(
            user_email="user@work.com",
            reply_history={"sender@x.com"},
        )
        # Known contact with same domain — many negative signals
        msg = _make_msg(sender="sender@work.com")
        score = classifier.spam_score(msg)
        assert 0.0 <= score <= 1.0

    def test_all_caps_subject_increases_score(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(subject="FREE MONEY NOW ACT FAST")
        score = classifier.spam_score(msg)
        normal_msg = _make_msg(subject="Meeting tomorrow")
        normal_score = classifier.spam_score(normal_msg)
        assert score > normal_score

    def test_cc_recipient_increases_score(self):
        classifier = EmailClassifier(user_email="user@work.com")
        # User is in CC, not direct To
        msg = _make_msg(recipients=["other@work.com", "user@work.com"])
        # Simulate: user is not the primary To recipient
        score_cc = classifier.spam_score(msg, user_is_direct=False)
        score_direct = classifier.spam_score(msg, user_is_direct=True)
        assert score_cc >= score_direct


class TestPriorityScoring:
    def test_known_contact_with_action_words_is_high(self):
        classifier = EmailClassifier(
            user_email="user@work.com",
            reply_history={"boss@work.com"},
        )
        msg = _make_msg(
            sender="boss@work.com",
            subject="Deadline moved to Friday — urgent",
        )
        priority = classifier.priority_score(msg)
        assert priority == "high"

    def test_unknown_direct_sender_is_medium(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(sender="newperson@other.com")
        priority = classifier.priority_score(msg)
        assert priority == "medium"

    def test_mailing_list_is_low(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            sender="noreply@newsletter.com",
            subject="Weekly digest",
        )
        priority = classifier.priority_score(msg, headers={"List-Unsubscribe": "yes"})
        assert priority == "low"


class TestCategoryDetection:
    def test_bill_detected(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            subject="Invoice #4521 — Payment Due March 20",
            snippet="Amount due: $142.50",
        )
        categories = classifier.detect_categories(msg)
        assert "bill" in categories

    def test_newsletter_detected(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            sender="news@techblog.com",
            subject="This Week in Tech — Issue #312",
        )
        categories = classifier.detect_categories(msg, headers={"List-Unsubscribe": "yes"})
        assert "newsletter" in categories

    def test_social_detected(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(
            sender="notifications@linkedin.com",
            subject="John viewed your profile",
        )
        categories = classifier.detect_categories(msg)
        assert "social" in categories

    def test_work_detected(self):
        classifier = EmailClassifier(user_email="user@work.com")
        msg = _make_msg(sender="colleague@work.com", subject="Q3 Report")
        categories = classifier.detect_categories(msg)
        assert "work" in categories
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_classifier.py -v`
Expected: ImportError — `EmailClassifier` not defined.

- [ ] **Step 3: Implement the classifier**

Create `src/homie_core/email/classifier.py`:
```python
"""Email spam scoring, priority scoring, and category detection.

Uses weighted heuristic signals — no external ML dependencies.
Scores are clamped to [0.0, 1.0].
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from homie_core.email.models import EmailMessage

# Spam phrases (lowercase)
_SPAM_PHRASES = {
    "act now", "click here", "limited time", "free gift", "winner",
    "congratulations", "claim your", "unsubscribe", "opt out",
}

# Action keywords for priority (lowercase)
_ACTION_KEYWORDS = {
    "urgent", "deadline", "asap", "payment", "meeting", "rsvp",
    "action required", "due date", "respond", "immediately",
    "important", "critical", "overdue",
}

# Social media senders (partial domain match)
_SOCIAL_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "instagram.com",
    "tiktok.com", "reddit.com", "quora.com", "pinterest.com",
    "facebookmail.com", "x.com",
}

# Bill/invoice patterns
_BILL_PATTERN = re.compile(
    r"(invoice|payment due|amount due|billing|statement|receipt|\$\d+|\€\d+|₹\d+)",
    re.IGNORECASE,
)


def _extract_domain(email_str: str) -> str:
    """Extract domain from 'Name <user@domain>' or 'user@domain'."""
    match = re.search(r"@([\w.-]+)", email_str)
    return match.group(1).lower() if match else ""


def _extract_email(email_str: str) -> str:
    """Extract bare email from 'Name <user@domain>' or 'user@domain'."""
    match = re.search(r"[\w.+-]+@[\w.-]+", email_str)
    return match.group(0).lower() if match else email_str.lower()


class EmailClassifier:
    """Heuristic email classifier for spam, priority, and categories."""

    def __init__(
        self,
        user_email: str,
        reply_history: set[str] | None = None,
        sender_corrections: dict[str, float] | None = None,
    ):
        self._user_email = user_email.lower()
        self._user_domain = _extract_domain(user_email)
        self._reply_history = {e.lower() for e in (reply_history or set())}
        self._sender_corrections = sender_corrections or {}

    def spam_score(
        self,
        msg: EmailMessage,
        headers: dict[str, str] | None = None,
        user_is_direct: bool = True,
    ) -> float:
        """Compute spam score in [0.0, 1.0]. Higher = more likely spam."""
        headers = headers or {}
        score = 0.0
        sender_email = _extract_email(msg.sender)
        sender_domain = _extract_domain(msg.sender)

        # Positive signals (increase spam likelihood)
        if sender_email not in self._reply_history:
            score += 0.3

        if headers.get("List-Unsubscribe"):
            score += 0.2

        if headers.get("Precedence", "").lower() == "bulk":
            score += 0.2

        # Subject analysis
        subject_upper_ratio = sum(1 for c in msg.subject if c.isupper()) / max(len(msg.subject), 1)
        if subject_upper_ratio > 0.6 and len(msg.subject) > 5:
            score += 0.2

        excessive_punct = len(re.findall(r"[!?]{2,}", msg.subject))
        if excessive_punct > 0:
            score += 0.1

        text = (msg.subject + " " + msg.snippet).lower()
        spam_hit = any(phrase in text for phrase in _SPAM_PHRASES)
        if spam_hit:
            score += 0.1

        # Negative signals (decrease spam likelihood)
        if sender_email in self._reply_history:
            score -= 0.5

        if user_is_direct:
            score -= 0.2

        if sender_domain == self._user_domain and self._user_domain:
            score -= 0.3

        # Per-sender correction from learning
        if sender_email in self._sender_corrections:
            score += self._sender_corrections[sender_email]

        return max(0.0, min(1.0, score))

    def priority_score(
        self,
        msg: EmailMessage,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Classify priority as 'high', 'medium', or 'low'."""
        headers = headers or {}
        sender_email = _extract_email(msg.sender)
        sender_domain = _extract_domain(msg.sender)
        text = (msg.subject + " " + msg.snippet).lower()

        is_known = sender_email in self._reply_history
        has_action = any(kw in text for kw in _ACTION_KEYWORDS)
        is_list = bool(headers.get("List-Unsubscribe"))
        is_social = any(d in sender_domain for d in _SOCIAL_DOMAINS)

        # High: known contact + action words, or same domain + action
        if (is_known and has_action) or (sender_domain == self._user_domain and has_action):
            return "high"

        # Low: mailing lists, social, automated
        if is_list or is_social:
            return "low"

        # Medium: everything else (direct unknown sender, etc.)
        return "medium"

    def detect_categories(
        self,
        msg: EmailMessage,
        headers: dict[str, str] | None = None,
    ) -> list[str]:
        """Detect content categories for auto-labeling."""
        headers = headers or {}
        categories = []
        sender_domain = _extract_domain(msg.sender)
        text = (msg.subject + " " + (msg.snippet or "")).lower()

        # Bill detection
        if _BILL_PATTERN.search(text):
            categories.append("bill")

        # Work: same domain
        if sender_domain == self._user_domain and self._user_domain:
            categories.append("work")

        # Newsletter
        if headers.get("List-Unsubscribe") and not any(d in sender_domain for d in _SOCIAL_DOMAINS):
            categories.append("newsletter")

        # Social
        if any(d in sender_domain for d in _SOCIAL_DOMAINS):
            categories.append("social")

        return categories
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_classifier.py -v`
Expected: All 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/classifier.py tests/unit/test_email/test_classifier.py
git commit -m "feat(email): add heuristic spam/priority classifier and category detection"
```

---

### Task 5: Organizer (`organizer.py`)

**Files:**
- Create: `src/homie_core/email/organizer.py`
- Create: `tests/unit/test_email/test_organizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_email/test_organizer.py`:
```python
"""Tests for email organizer — labeling, archiving, financial extraction."""
from __future__ import annotations

import re
from unittest.mock import MagicMock, call

from homie_core.email.organizer import EmailOrganizer
from homie_core.email.models import EmailMessage, EmailSyncConfig


def _make_msg(**overrides) -> EmailMessage:
    defaults = dict(
        id="msg1", thread_id="t1", account_id="user@work.com",
        provider="gmail", subject="Hello", sender="alice@example.com",
        recipients=["user@work.com"], snippet="Hey there...",
        priority="medium", spam_score=0.0, categories=[],
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


class TestLabelApplication:
    def test_bill_gets_homie_bills_label(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={"bill": "Label_Bills"})
        msg = _make_msg(categories=["bill"])
        organizer.apply_labels(msg)
        provider.apply_label.assert_called_with("msg1", "Label_Bills")

    def test_multiple_categories_get_multiple_labels(self):
        provider = MagicMock()
        organizer = EmailOrganizer(
            provider=provider,
            label_ids={"bill": "L_B", "work": "L_W"},
        )
        msg = _make_msg(categories=["bill", "work"])
        organizer.apply_labels(msg)
        assert provider.apply_label.call_count == 2

    def test_no_categories_no_labels(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(categories=[])
        organizer.apply_labels(msg)
        provider.apply_label.assert_not_called()


class TestArchiveRules:
    def test_low_priority_not_direct_archived(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(priority="low", recipients=["other@x.com", "user@work.com"])
        result = organizer.should_archive(msg, user_is_direct=False)
        assert result is True

    def test_high_priority_not_archived(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(priority="high")
        result = organizer.should_archive(msg, user_is_direct=True)
        assert result is False

    def test_social_archived(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(categories=["social"])
        result = organizer.should_archive(msg, user_is_direct=True)
        assert result is True

    def test_spam_above_threshold_trashed(self):
        provider = MagicMock()
        config = EmailSyncConfig(account_id="user@work.com", auto_trash_spam=True)
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(spam_score=0.85)
        result = organizer.should_trash(msg, config)
        assert result is True

    def test_spam_below_threshold_not_trashed(self):
        provider = MagicMock()
        config = EmailSyncConfig(account_id="user@work.com", auto_trash_spam=True)
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(spam_score=0.5)
        result = organizer.should_trash(msg, config)
        assert result is False

    def test_auto_trash_disabled(self):
        provider = MagicMock()
        config = EmailSyncConfig(account_id="user@work.com", auto_trash_spam=False)
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(spam_score=0.9)
        result = organizer.should_trash(msg, config)
        assert result is False


class TestFinancialExtraction:
    def test_extract_amount_and_date(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(
            subject="Invoice #4521 — Payment Due March 20",
            snippet="Amount due: $142.50. Please pay by March 20, 2026.",
        )
        result = organizer.extract_financial(msg)
        assert result is not None
        assert result["amount"] == "142.50"
        assert result["currency"] == "USD"

    def test_no_financial_data(self):
        provider = MagicMock()
        organizer = EmailOrganizer(provider=provider, label_ids={})
        msg = _make_msg(subject="Hey", snippet="How are you?")
        result = organizer.extract_financial(msg)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_organizer.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the organizer**

Create `src/homie_core/email/organizer.py`:
```python
"""Email organizer — auto-labeling, archiving, and financial extraction.

Works with any EmailProvider to apply labels and archive/trash messages
based on classifier output.
"""
from __future__ import annotations

import re
from typing import Any

from homie_core.email.models import EmailMessage, EmailSyncConfig
from homie_core.email.provider import EmailProvider

# Category → Homie label name mapping
HOMIE_LABELS = {
    "bill": "Homie/Bills",
    "work": "Homie/Work",
    "newsletter": "Homie/Newsletters",
    "social": "Homie/Social",
    "review": "Homie/Review",
}

# Currency patterns
_CURRENCY_PATTERNS = [
    (re.compile(r"\$\s*([\d,]+\.?\d*)"), "USD"),
    (re.compile(r"€\s*([\d,]+\.?\d*)"), "EUR"),
    (re.compile(r"£\s*([\d,]+\.?\d*)"), "GBP"),
    (re.compile(r"₹\s*([\d,]+\.?\d*)"), "INR"),
]


class EmailOrganizer:
    """Applies labels, decides archiving, and extracts financial data."""

    def __init__(
        self,
        provider: EmailProvider,
        label_ids: dict[str, str],
    ):
        self._provider = provider
        self._label_ids = label_ids  # category -> Gmail label ID

    def apply_labels(self, msg: EmailMessage) -> None:
        """Apply Homie labels based on message categories."""
        for category in msg.categories:
            label_id = self._label_ids.get(category)
            if label_id:
                self._provider.apply_label(msg.id, label_id)

    def should_archive(self, msg: EmailMessage, user_is_direct: bool = True,
                       sender_open_count: int = 999) -> bool:
        """Decide whether to archive (remove from inbox).

        Args:
            sender_open_count: Number of last N emails from this sender that user opened.
                               Default 999 (don't archive). Pass actual count for newsletters.
        """
        # Social always archived
        if "social" in msg.categories:
            return True
        # Low priority + not direct recipient
        if msg.priority == "low" and not user_is_direct:
            return True
        # Newsletter + user hasn't opened last 3 from same sender (spec Section 8.2)
        if "newsletter" in msg.categories and sender_open_count < 3:
            return True
        return False

    def should_trash(self, msg: EmailMessage, config: EmailSyncConfig) -> bool:
        """Decide whether to trash (spam score > 0.8 and auto_trash enabled)."""
        if not config.auto_trash_spam:
            return False
        return msg.spam_score > 0.8

    def extract_financial(self, msg: EmailMessage) -> dict[str, Any] | None:
        """Extract financial data (amount, currency) from bill emails.

        Returns dict with keys: amount, currency, description, source.
        Returns None if no financial data found.
        """
        text = f"{msg.subject} {msg.snippet}"

        amount = None
        currency = None
        for pattern, curr in _CURRENCY_PATTERNS:
            match = pattern.search(text)
            if match:
                amount = match.group(1).replace(",", "")
                currency = curr
                break

        if amount is None:
            return None

        return {
            "source": f"gmail:{msg.id}",
            "category": "bill",
            "description": msg.subject,
            "amount": amount,
            "currency": currency,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_organizer.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/organizer.py tests/unit/test_email/test_organizer.py
git commit -m "feat(email): add organizer — labeling, archive rules, financial extraction"
```
## Chunk 3: OAuth + Gmail Provider (Tasks 6-7)

### Task 6: OAuth Flow (`oauth.py`)

**Files:**
- Create: `src/homie_core/email/oauth.py`
- Create: `tests/unit/test_email/test_oauth.py`

- [ ] **Step 1: Write failing tests for OAuth**

Create `tests/unit/test_email/test_oauth.py`:
```python
"""Tests for OAuth 2.0 flow — local redirect + manual fallback."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

from homie_core.email.oauth import (
    GmailOAuth,
    GMAIL_SCOPES,
    build_auth_url,
    exchange_code,
)


class TestBuildAuthUrl:
    def test_returns_url_with_scopes(self):
        url = build_auth_url(client_id="test-id", redirect_uri="http://localhost:8547/callback")
        assert "test-id" in url
        assert "scope=" in url
        assert "localhost" in url

    def test_includes_all_scopes(self):
        url = build_auth_url(client_id="test-id", redirect_uri="urn:ietf:wg:oauth:2.0:oob")
        for scope in GMAIL_SCOPES:
            # URL-encoded scope
            assert "gmail" in url


class TestExchangeCode:
    @patch("homie_core.email.oauth.requests.post")
    def test_exchange_returns_tokens(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "ya29.xxx",
            "refresh_token": "1//xxx",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_resp

        tokens = exchange_code(
            code="auth-code",
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="http://localhost:8547/callback",
        )
        assert tokens["access_token"] == "ya29.xxx"
        assert tokens["refresh_token"] == "1//xxx"

    @patch("homie_core.email.oauth.requests.post")
    def test_exchange_error_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "invalid_grant"}
        mock_resp.raise_for_status.side_effect = Exception("400 Bad Request")
        mock_post.return_value = mock_resp

        try:
            exchange_code("bad-code", "id", "secret", "uri")
            assert False, "Should have raised"
        except Exception:
            pass


class TestGmailOAuth:
    def test_scopes_defined(self):
        assert len(GMAIL_SCOPES) == 3
        assert "https://www.googleapis.com/auth/gmail.readonly" in GMAIL_SCOPES

    @patch("homie_core.email.oauth.requests.post")
    def test_refresh_token(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "ya29.new",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_resp

        oauth = GmailOAuth(client_id="cid", client_secret="csec")
        result = oauth.refresh_access_token("1//refresh")
        assert result["access_token"] == "ya29.new"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_oauth.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement OAuth flow**

Create `src/homie_core/email/oauth.py`:
```python
"""OAuth 2.0 flow for Gmail — local redirect server + manual fallback.

Handles:
1. Building the authorization URL
2. Running a local HTTP server to receive the redirect (port 8547)
3. Manual code entry fallback for headless environments
4. Code-to-token exchange
5. Token refresh
"""
from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.parse
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
]

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_REDIRECT_PORT = 8547
_REDIRECT_URI = f"http://localhost:{_REDIRECT_PORT}/callback"
_MANUAL_REDIRECT_URI = "urn:ietf:wg:oauth:2.0:oob"


def build_auth_url(
    client_id: str,
    redirect_uri: str = _REDIRECT_URI,
) -> str:
    """Build the Google OAuth consent screen URL."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(GMAIL_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str = _REDIRECT_URI,
) -> dict[str, Any]:
    """Exchange authorization code for access + refresh tokens."""
    if requests is None:
        raise ImportError("requests library required for OAuth")

    resp = requests.post(_GOOGLE_TOKEN_URL, data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler to capture OAuth redirect callback."""

    auth_code: str | None = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            _CallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Authorization successful!</h2>"
                             b"<p>You can close this tab and return to Homie.</p></body></html>")
        else:
            error = params.get("error", ["unknown"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body><h2>Error: {error}</h2></body></html>".encode())

    def log_message(self, format, *args):
        pass  # Suppress server logs


class GmailOAuth:
    """Handles Gmail OAuth 2.0 lifecycle."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret

    def get_auth_url(self, use_local_server: bool = True) -> str:
        """Get the authorization URL."""
        redirect = _REDIRECT_URI if use_local_server else _MANUAL_REDIRECT_URI
        return build_auth_url(self._client_id, redirect)

    def wait_for_redirect(self, timeout: int = 120) -> str | None:
        """Start local server and wait for OAuth redirect. Returns auth code or None."""
        _CallbackHandler.auth_code = None
        try:
            server = http.server.HTTPServer(("localhost", _REDIRECT_PORT), _CallbackHandler)
        except OSError:
            return None  # Port unavailable

        server.timeout = timeout
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        thread.join(timeout=timeout + 5)
        server.server_close()
        return _CallbackHandler.auth_code

    def exchange(self, code: str, use_local_server: bool = True) -> dict[str, Any]:
        """Exchange auth code for tokens."""
        redirect = _REDIRECT_URI if use_local_server else _MANUAL_REDIRECT_URI
        return exchange_code(code, self._client_id, self._client_secret, redirect)

    def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired access token."""
        if requests is None:
            raise ImportError("requests library required for OAuth")

        resp = requests.post(_GOOGLE_TOKEN_URL, data={
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_oauth.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/oauth.py tests/unit/test_email/test_oauth.py
git commit -m "feat(email): add OAuth 2.0 flow with local redirect + manual fallback"
```

---

### Task 7: Gmail Provider (`gmail_provider.py`)

**Files:**
- Create: `src/homie_core/email/gmail_provider.py`
- Create: `tests/unit/test_email/test_gmail_provider.py`

- [ ] **Step 1: Write failing tests for GmailProvider**

Create `tests/unit/test_email/test_gmail_provider.py`:
```python
"""Tests for Gmail provider — all API calls mocked."""
from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock, patch, PropertyMock

from homie_core.email.gmail_provider import GmailProvider
from homie_core.email.models import EmailMessage, Label


def _mock_service():
    """Create a mock Gmail API service object."""
    return MagicMock()


def _make_gmail_message(msg_id="msg1", thread_id="t1", subject="Test",
                         sender="alice@x.com", to="user@gmail.com",
                         snippet="Hello...", date_ms=1710288000000,
                         labels=None, has_body=True):
    """Build a Gmail API message response dict."""
    headers = [
        {"name": "Subject", "value": subject},
        {"name": "From", "value": sender},
        {"name": "To", "value": to},
        {"name": "Date", "value": "Tue, 12 Mar 2024 12:00:00 +0000"},
    ]
    msg = {
        "id": msg_id,
        "threadId": thread_id,
        "snippet": snippet,
        "labelIds": labels or ["INBOX"],
        "internalDate": str(date_ms),
        "payload": {
            "headers": headers,
            "mimeType": "text/plain",
        },
    }
    if has_body:
        msg["payload"]["body"] = {
            "data": base64.urlsafe_b64encode(b"Hello body").decode()
        }
    return msg


class TestGmailProviderParsing:
    def test_parse_message(self):
        provider = GmailProvider(account_id="user@gmail.com")
        raw = _make_gmail_message()
        msg = provider._parse_message(raw)
        assert msg.id == "msg1"
        assert msg.subject == "Test"
        assert msg.sender == "alice@x.com"
        assert msg.provider == "gmail"

    def test_parse_message_with_attachments(self):
        provider = GmailProvider(account_id="user@gmail.com")
        raw = _make_gmail_message()
        raw["payload"]["parts"] = [
            {"filename": "doc.pdf", "mimeType": "application/pdf", "body": {"size": 1024}},
        ]
        msg = provider._parse_message(raw)
        assert msg.has_attachments is True
        assert "doc.pdf" in msg.attachment_names


class TestGmailProviderFetch:
    def test_fetch_messages(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        # Mock messages().list()
        service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}, {"id": "msg2"}],
        }
        # Mock messages().get()
        service.users().messages().get().execute.side_effect = [
            _make_gmail_message(msg_id="msg1"),
            _make_gmail_message(msg_id="msg2", subject="Second"),
        ]

        messages = provider.fetch_messages(since=0.0, max_results=10)
        assert len(messages) == 2

    def test_search(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg1"}],
        }
        service.users().messages().get().execute.return_value = _make_gmail_message()

        results = provider.search("from:alice")
        assert len(results) == 1


class TestGmailProviderActions:
    def test_apply_label(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        provider.apply_label("msg1", "Label_1")
        service.users().messages().modify.assert_called()

    def test_trash(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service

        provider.trash("msg1")
        service.users().messages().trash.assert_called()

    def test_create_draft(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().drafts().create().execute.return_value = {"id": "draft1"}

        draft_id = provider.create_draft(to="bob@x.com", subject="Re: Hi", body="Hello Bob")
        assert draft_id == "draft1"

    def test_get_profile(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().getProfile().execute.return_value = {
            "emailAddress": "user@gmail.com",
        }

        profile = provider.get_profile()
        assert profile["emailAddress"] == "user@gmail.com"

    def test_list_labels(self):
        provider = GmailProvider(account_id="user@gmail.com")
        service = _mock_service()
        provider._service = service
        service.users().labels().list().execute.return_value = {
            "labels": [
                {"id": "INBOX", "name": "INBOX", "type": "system"},
                {"id": "Label_1", "name": "Homie/Bills", "type": "user"},
            ],
        }

        labels = provider.list_labels()
        assert len(labels) == 2
        assert labels[0].id == "INBOX"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_gmail_provider.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement GmailProvider**

Create `src/homie_core/email/gmail_provider.py`:
```python
"""Gmail API implementation of EmailProvider.

Uses google-api-python-client to interact with Gmail.
All API calls go through self._service (a googleapiclient Resource).
"""
from __future__ import annotations

import base64
import email.mime.text
import time
from typing import Any

from homie_core.email.models import EmailMessage, HistoryChange, Label
from homie_core.email.provider import EmailProvider


def _header(headers: list[dict], name: str) -> str:
    """Extract a header value from Gmail API headers list."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _decode_body(payload: dict) -> str:
    """Decode message body from Gmail API payload."""
    # Direct body
    body_data = payload.get("body", {}).get("data", "")
    if body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    # Multipart — find text/plain
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        # Nested multipart
        if part.get("parts"):
            result = _decode_body(part)
            if result:
                return result

    return ""


class GmailProvider(EmailProvider):
    """Gmail API implementation."""

    def __init__(self, account_id: str):
        self._account_id = account_id
        self._service = None  # Set by authenticate() or test injection

    def authenticate(self, credential, vault=None, client_id: str = "", client_secret: str = "") -> None:
        """Build Gmail API service from stored Credential dataclass.

        Args:
            credential: vault Credential dataclass (attribute access: .access_token, etc.)
            vault: SecureVault instance for token refresh and revocation handling
            client_id: OAuth client ID (for token refresh)
            client_secret: OAuth client secret (for token refresh)
        """
        self._vault = vault
        self._credential_id = f"{credential.provider}:{credential.account_id}"
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = Credentials(
                token=credential.access_token,
                refresh_token=credential.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
            )
            self._creds = creds
            self._service = build("gmail", "v1", credentials=creds)
        except ImportError:
            raise ImportError(
                "Gmail provider requires google-api-python-client. "
                "Install with: pip install 'homie-ai[email]'"
            )

    def _check_token_freshness(self) -> None:
        """Check and refresh token if expired. Handles revocation.

        Per spec Section 5.3: Before any API call, check expires_at.
        If refresh fails → mark disconnected, log consent, notify user.
        """
        if not self._creds or not self._creds.expired:
            return
        try:
            from google.auth.transport.requests import Request
            self._creds.refresh(Request())
            if self._vault:
                self._vault.refresh_credential(
                    self._credential_id,
                    new_access_token=self._creds.token,
                    new_expires_at=self._creds.expiry.timestamp() if self._creds.expiry else None,
                )
        except Exception:
            if self._vault:
                self._vault.set_connection_status("gmail", connected=False)
                self._vault.log_consent("gmail", "token_revoked", reason="refresh_failed")
            raise

    def fetch_messages(self, since: float, max_results: int = 100) -> list[EmailMessage]:
        """Fetch messages newer than `since` timestamp."""
        self._check_token_freshness()
        query = f"newer_than:7d" if since == 0.0 else f"after:{int(since)}"
        response = (
            self._service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        msg_ids = [m["id"] for m in response.get("messages", [])]
        return [self._fetch_and_parse(mid) for mid in msg_ids]

    def fetch_message_body(self, message_id: str) -> str:
        """Fetch full body text of a specific message."""
        self._check_token_freshness()
        raw = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return _decode_body(raw.get("payload", {}))

    def get_history(self, start_history_id: str) -> tuple[list[HistoryChange], str]:
        """Get changes since history_id."""
        self._check_token_freshness()
        changes = []
        response = (
            self._service.users()
            .history()
            .list(userId="me", startHistoryId=start_history_id)
            .execute()
        )
        new_history_id = response.get("historyId", start_history_id)

        for record in response.get("history", []):
            for added in record.get("messagesAdded", []):
                changes.append(HistoryChange(
                    message_id=added["message"]["id"],
                    change_type="added",
                    labels=added["message"].get("labelIds", []),
                ))
            for deleted in record.get("messagesDeleted", []):
                changes.append(HistoryChange(
                    message_id=deleted["message"]["id"],
                    change_type="deleted",
                ))
            for label_added in record.get("labelsAdded", []):
                changes.append(HistoryChange(
                    message_id=label_added["message"]["id"],
                    change_type="labelAdded",
                    labels=label_added.get("labelIds", []),
                ))
            for label_removed in record.get("labelsRemoved", []):
                changes.append(HistoryChange(
                    message_id=label_removed["message"]["id"],
                    change_type="labelRemoved",
                    labels=label_removed.get("labelIds", []),
                ))

        return changes, new_history_id

    def search(self, query: str, max_results: int = 20) -> list[EmailMessage]:
        """Search using Gmail query syntax."""
        self._check_token_freshness()
        response = (
            self._service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        msg_ids = [m["id"] for m in response.get("messages", [])]
        return [self._fetch_and_parse(mid) for mid in msg_ids]

    def apply_label(self, message_id: str, label_id: str) -> None:
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"addLabelIds": [label_id]},
        ).execute()

    def remove_label(self, message_id: str, label_id: str) -> None:
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": [label_id]},
        ).execute()

    def trash(self, message_id: str) -> None:
        self._service.users().messages().trash(userId="me", id=message_id).execute()

    def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        reply_to: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> str:
        """Create a draft. Returns draft ID."""
        mime_msg = email.mime.text.MIMEText(body)
        mime_msg["To"] = to
        mime_msg["Subject"] = subject
        if cc:
            mime_msg["Cc"] = ", ".join(cc)
        if bcc:
            mime_msg["Bcc"] = ", ".join(bcc)
        if reply_to:
            mime_msg["In-Reply-To"] = reply_to
            mime_msg["References"] = reply_to

        encoded = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        draft_body: dict[str, Any] = {"message": {"raw": encoded}}

        result = (
            self._service.users()
            .drafts()
            .create(userId="me", body=draft_body)
            .execute()
        )
        return result["id"]

    def list_labels(self) -> list[Label]:
        response = self._service.users().labels().list(userId="me").execute()
        return [
            Label(id=l["id"], name=l["name"], type=l.get("type", "user"))
            for l in response.get("labels", [])
        ]

    def get_profile(self) -> dict:
        return self._service.users().getProfile(userId="me").execute()

    def mark_read(self, message_id: str) -> None:
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()

    def archive(self, message_id: str) -> None:
        self._service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": ["INBOX"]},
        ).execute()

    # --- Internal helpers ---

    def _fetch_and_parse(self, message_id: str) -> EmailMessage:
        """Fetch a single message by ID and parse it."""
        raw = (
            self._service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        return self._parse_message(raw)

    def _parse_message(self, raw: dict) -> EmailMessage:
        """Parse a Gmail API message dict into EmailMessage."""
        payload = raw.get("payload", {})
        headers = payload.get("headers", [])

        # Check for attachments
        attachment_names = []
        has_attachments = False
        for part in payload.get("parts", []):
            filename = part.get("filename", "")
            if filename:
                has_attachments = True
                attachment_names.append(filename)

        date_ms = int(raw.get("internalDate", "0"))
        label_ids = raw.get("labelIds", [])

        return EmailMessage(
            id=raw["id"],
            thread_id=raw.get("threadId", ""),
            account_id=self._account_id,
            provider="gmail",
            subject=_header(headers, "Subject"),
            sender=_header(headers, "From"),
            recipients=(
                [r.strip() for r in _header(headers, "To").split(",") if r.strip()]
                + [r.strip() for r in _header(headers, "Cc").split(",") if r.strip()]
            ),
            snippet=raw.get("snippet", ""),
            body=None,  # Body fetched separately on demand
            labels=label_ids,
            date=date_ms / 1000.0,
            is_read="UNREAD" not in label_ids,
            is_starred="STARRED" in label_ids,
            has_attachments=has_attachments,
            attachment_names=attachment_names,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_gmail_provider.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/gmail_provider.py tests/unit/test_email/test_gmail_provider.py
git commit -m "feat(email): add GmailProvider — Gmail API implementation of EmailProvider"
```
## Chunk 4: Sync Engine + AI Tools (Tasks 8-9)

### Task 8: Sync Engine (`sync_engine.py`)

**Files:**
- Create: `src/homie_core/email/sync_engine.py`
- Create: `tests/unit/test_email/test_sync_engine.py`

- [ ] **Step 1: Write failing tests for SyncEngine**

Create `tests/unit/test_email/test_sync_engine.py`:
```python
"""Tests for email sync engine — initial + incremental sync."""
from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock, patch

from homie_core.email.sync_engine import SyncEngine
from homie_core.email.models import (
    EmailMessage, EmailSyncConfig, HistoryChange, SyncResult, SyncState,
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

        # Mock the single message fetch
        def mock_fetch_and_parse(msg_id):
            if msg_id == "m3":
                return new_msg
            return None
        provider._fetch_and_parse = mock_fetch_and_parse

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_sync_engine.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the sync engine**

Create `src/homie_core/email/sync_engine.py`:
```python
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
from typing import Any

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
    """Manages email sync lifecycle for one account.

    Coordinates: provider (fetch) → classifier (score) → organizer (label/archive)
    → store in cache.db → notify via working_memory.
    """

    def __init__(
        self,
        provider: EmailProvider,
        classifier: EmailClassifier,
        cache_conn: sqlite3.Connection,
        account_id: str,
        organizer=None,         # EmailOrganizer (optional, for label/archive/financial)
        vault=None,             # SecureVault (optional, for store_financial)
        working_memory=None,    # WorkingMemory (optional, for notifications)
    ):
        self._provider = provider
        self._classifier = classifier
        self._conn = cache_conn
        self._account_id = account_id
        self._organizer = organizer
        self._vault = vault
        self._working_memory = working_memory

    def initial_sync(self) -> SyncResult:
        """Full sync — fetch last 7 days, classify, organize, store."""
        result = SyncResult(account_id=self._account_id)

        try:
            messages = self._provider.fetch_messages(since=0.0, max_results=100)
            profile = self._provider.get_profile()
            history_id = profile.get("historyId", "")

            for msg in messages:
                self._classify_and_organize(msg, result, config=None)

            # Evict emails older than 90 days (spec Section 11)
            self._evict_old_emails()

            # Save sync state
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
        """Incremental sync via historyId delta.

        Error handling per spec Section 6.4:
        - HttpError 429: skip tick, retry next cycle
        - HttpError 401: refresh token, mark errored if fails
        - HttpError 403: mark disconnected, log consent
        - Network error: skip tick, retry next cycle
        """
        result = SyncResult(account_id=self._account_id)

        # Load sync state
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
                        msg = self._provider._fetch_and_parse(change.message_id)
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
                    # Detect spam corrections (spec Section 7.4)
                    self._check_spam_correction(change)
                    self._conn.execute(
                        "UPDATE emails SET labels=? WHERE id=? AND account_id=?",
                        (json.dumps(change.labels), change.message_id, self._account_id),
                    )
                    result.updated_messages += 1

            # Update sync state
            self._conn.execute(
                """UPDATE email_sync_state
                   SET history_id=?, last_incremental_sync=?,
                       total_synced=total_synced+?
                   WHERE account_id=?""",
                (new_history_id, time.time(), result.new_messages, self._account_id),
            )
            self._conn.commit()

        except Exception as e:
            # Spec Section 6.4: categorize errors
            error_str = str(e)
            if "429" in error_str or "HttpError 429" in error_str:
                result.errors.append("rate_limited")  # Skip tick, retry next cycle
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
        # Classify
        msg.spam_score = self._classifier.spam_score(msg)
        msg.priority = self._classifier.priority_score(msg)
        msg.categories = self._classifier.detect_categories(msg)

        # Add "review" category for borderline spam (spec Section 7.1)
        if 0.3 <= msg.spam_score <= 0.8 and "review" not in msg.categories:
            msg.categories.append("review")

        # Organize: apply labels, archive/trash (spec Section 8)
        if self._organizer:
            try:
                self._organizer.apply_labels(msg)
                if config and self._organizer.should_trash(msg, config):
                    self._provider.trash(msg.id)
                    result.trashed_messages += 1
                    return  # Don't store trashed messages
                # Query open count for newsletters (spec Section 8.2)
                open_count = 999
                if "newsletter" in msg.categories:
                    row = self._conn.execute(
                        "SELECT COUNT(*) FROM emails WHERE sender=? AND account_id=? AND is_read=1 "
                        "ORDER BY date DESC LIMIT 3",
                        (msg.sender, self._account_id),
                    ).fetchone()
                    open_count = row[0] if row else 0
                if self._organizer.should_archive(msg, sender_open_count=open_count):
                    self._provider.archive(msg.id)
            except Exception:
                pass

            # Financial extraction (spec Section 8.3)
            if "bill" in msg.categories and self._vault:
                try:
                    fin_data = self._organizer.extract_financial(msg)
                    if fin_data:
                        self._vault.store_financial(**fin_data)
                except Exception:
                    pass

        # Store in cache
        self._store_message(msg)
        result.new_messages += 1

        # Notification (spec Section 10.2)
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
        """Remove emails older than 90 days from cache (spec Section 11)."""
        cutoff = time.time() - (90 * 86400)
        self._conn.execute(
            "DELETE FROM emails WHERE account_id=? AND date < ?",
            (self._account_id, cutoff),
        )

    def _check_spam_correction(self, change: HistoryChange) -> None:
        """Detect spam corrections from label changes (spec Section 7.4).

        If user un-trashes (TRASH removed) or manually trashes (TRASH added),
        store correction in spam_corrections table.
        """
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

    def _classify_message(self, msg: EmailMessage) -> None:
        """Run classifier on a message (mutates in place)."""
        msg.spam_score = self._classifier.spam_score(msg)
        msg.priority = self._classifier.priority_score(msg)
        msg.categories = self._classifier.detect_categories(msg)

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

        # Never notify for suspicious emails
        if msg.spam_score >= 0.3:
            return False

        # Check priority threshold
        priority_order = {"high": 3, "medium": 2, "low": 1, "all": 0}
        msg_level = priority_order.get(msg.priority, 1)
        threshold = priority_order.get(config.notify_priority, 3)
        if msg_level < threshold:
            return False

        # Check quiet hours
        if config.quiet_hours_start is not None and config.quiet_hours_end is not None:
            hour = _current_hour()
            if config.quiet_hours_start > config.quiet_hours_end:
                # Wraps midnight: e.g., 22-7
                if hour >= config.quiet_hours_start or hour < config.quiet_hours_end:
                    return False
            else:
                if config.quiet_hours_start <= hour < config.quiet_hours_end:
                    return False

        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_sync_engine.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/sync_engine.py tests/unit/test_email/test_sync_engine.py
git commit -m "feat(email): add sync engine — initial + incremental sync with notifications"
```

---

### Task 9: AI Tools (`tools.py`)

**Files:**
- Create: `src/homie_core/email/tools.py`
- Create: `tests/unit/test_email/test_tools.py`

- [ ] **Step 1: Write failing tests for email tools**

Create `tests/unit/test_email/test_tools.py`:
```python
"""Tests for email AI tool wrappers."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from homie_core.brain.tool_registry import ToolRegistry
from homie_core.email.tools import register_email_tools
from homie_core.email.models import EmailMessage


def _make_msg(**overrides) -> EmailMessage:
    defaults = dict(
        id="msg1", thread_id="t1", account_id="user@gmail.com",
        provider="gmail", subject="Hello", sender="alice@x.com",
        recipients=["user@gmail.com"], snippet="Hey there...",
        priority="high", spam_score=0.0, is_read=False,
    )
    defaults.update(overrides)
    return EmailMessage(**defaults)


class TestEmailToolRegistration:
    def test_registers_9_tools(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        register_email_tools(registry, email_service)

        tool_names = {t.name for t in registry.list_tools()}
        expected = {
            "email_search", "email_read", "email_thread",
            "email_draft", "email_labels", "email_summary",
            "email_unread", "email_archive", "email_mark_read",
        }
        assert expected.issubset(tool_names)

    def test_all_tools_have_email_category(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        register_email_tools(registry, email_service)

        for tool in registry.list_tools():
            if tool.name.startswith("email_"):
                assert tool.category == "email"


class TestEmailSearchTool:
    def test_search_returns_json(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        email_service.search.return_value = [_make_msg()]
        register_email_tools(registry, email_service)

        tool = registry.get("email_search")
        result = tool.execute(query="from:alice")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["subject"] == "Hello"


class TestEmailReadTool:
    def test_read_returns_body(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        email_service.read_message.return_value = {
            "subject": "Hello",
            "sender": "alice@x.com",
            "body": "Full message body here",
        }
        register_email_tools(registry, email_service)

        tool = registry.get("email_read")
        result = tool.execute(message_id="msg1")
        data = json.loads(result)
        assert data["body"] == "Full message body here"


class TestEmailDraftTool:
    def test_draft_returns_id(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        email_service.create_draft.return_value = "draft_123"
        register_email_tools(registry, email_service)

        tool = registry.get("email_draft")
        result = tool.execute(to="bob@x.com", subject="Re: Hi", body="Thanks!")
        data = json.loads(result)
        assert data["draft_id"] == "draft_123"


class TestEmailUnreadTool:
    def test_unread_returns_grouped(self):
        registry = ToolRegistry()
        email_service = MagicMock()
        email_service.get_unread.return_value = {
            "high": [_make_msg().to_dict()],
            "medium": [],
            "low": [],
        }
        register_email_tools(registry, email_service)

        tool = registry.get("email_unread")
        result = tool.execute()
        data = json.loads(result)
        assert "high" in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_email/test_tools.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement email tools**

Create `src/homie_core/email/tools.py`:
```python
"""AI tool wrappers for email integration.

Registers 9 tools with the ToolRegistry so the Brain can interact
with email via the agentic loop.
"""
from __future__ import annotations

import json

from homie_core.brain.tool_registry import Tool, ToolParam, ToolRegistry

_MAX_OUTPUT = 2000


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "..."
    return text


def register_email_tools(registry: ToolRegistry, email_service) -> None:
    """Register all email tools with the tool registry."""

    def tool_email_search(query: str, account: str = "all", max_results: str = "10") -> str:
        results = email_service.search(query, account=account, max_results=int(max_results))
        return _truncate(json.dumps([m.to_dict() if hasattr(m, "to_dict") else m for m in results]))

    registry.register(Tool(
        name="email_search",
        description="Search emails using Gmail query syntax (e.g. 'from:alice subject:meeting').",
        params=[
            ToolParam(name="query", description="Gmail search query", type="string"),
            ToolParam(name="account", description="Account email or 'all'", type="string", required=False, default="all"),
            ToolParam(name="max_results", description="Maximum results", type="string", required=False, default="10"),
        ],
        execute=tool_email_search,
        category="email",
    ))

    def tool_email_read(message_id: str) -> str:
        result = email_service.read_message(message_id)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_read",
        description="Read the full body of a specific email by its message ID.",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_read,
        category="email",
    ))

    def tool_email_thread(thread_id: str) -> str:
        result = email_service.get_thread(thread_id)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_thread",
        description="Get all messages in a conversation thread.",
        params=[
            ToolParam(name="thread_id", description="Thread/conversation ID", type="string"),
        ],
        execute=tool_email_thread,
        category="email",
    ))

    def tool_email_draft(
        to: str, subject: str, body: str,
        reply_to: str = "", cc: str = "", bcc: str = "",
        account: str = "",
    ) -> str:
        draft_id = email_service.create_draft(
            to=to, subject=subject, body=body,
            reply_to=reply_to or None,
            cc=[c.strip() for c in cc.split(",") if c.strip()] if cc else None,
            bcc=[b.strip() for b in bcc.split(",") if b.strip()] if bcc else None,
            account=account or None,
        )
        return json.dumps({"draft_id": draft_id, "status": "Draft created. User must review and send manually."})

    registry.register(Tool(
        name="email_draft",
        description="Create a draft email (never sends automatically). User reviews and sends manually.",
        params=[
            ToolParam(name="to", description="Recipient email", type="string"),
            ToolParam(name="subject", description="Email subject", type="string"),
            ToolParam(name="body", description="Email body text", type="string"),
            ToolParam(name="reply_to", description="Message ID to reply to", type="string", required=False, default=""),
            ToolParam(name="cc", description="CC recipients (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="bcc", description="BCC recipients (comma-separated)", type="string", required=False, default=""),
            ToolParam(name="account", description="Send from this account", type="string", required=False, default=""),
        ],
        execute=tool_email_draft,
        category="email",
    ))

    def tool_email_labels(account: str = "") -> str:
        labels = email_service.list_labels(account=account or None)
        return json.dumps(labels)

    registry.register(Tool(
        name="email_labels",
        description="List all email labels/folders.",
        params=[
            ToolParam(name="account", description="Account email (optional)", type="string", required=False, default=""),
        ],
        execute=tool_email_labels,
        category="email",
    ))

    def tool_email_summary(days: str = "1") -> str:
        result = email_service.get_summary(days=int(days))
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_summary",
        description="Get email summary: unread count, high-priority items, action items.",
        params=[
            ToolParam(name="days", description="Number of days to summarize", type="string", required=False, default="1"),
        ],
        execute=tool_email_summary,
        category="email",
    ))

    def tool_email_unread(account: str = "all") -> str:
        result = email_service.get_unread(account=account)
        return _truncate(json.dumps(result))

    registry.register(Tool(
        name="email_unread",
        description="List unread emails grouped by priority (high, medium, low).",
        params=[
            ToolParam(name="account", description="Account email or 'all'", type="string", required=False, default="all"),
        ],
        execute=tool_email_unread,
        category="email",
    ))

    def tool_email_archive(message_id: str) -> str:
        email_service.archive_message(message_id)
        return json.dumps({"status": "archived", "message_id": message_id})

    registry.register(Tool(
        name="email_archive",
        description="Archive a message (remove from inbox).",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_archive,
        category="email",
    ))

    def tool_email_mark_read(message_id: str) -> str:
        email_service.mark_read(message_id)
        return json.dumps({"status": "marked_read", "message_id": message_id})

    registry.register(Tool(
        name="email_mark_read",
        description="Mark a message as read.",
        params=[
            ToolParam(name="message_id", description="Email message ID", type="string"),
        ],
        execute=tool_email_mark_read,
        category="email",
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_email/test_tools.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/email/tools.py tests/unit/test_email/test_tools.py
git commit -m "feat(email): add 9 AI tool wrappers for ToolRegistry"
```
## Chunk 5: Integration — CLI, Daemon, Dependencies (Tasks 10-12)

### Task 10: Update Dependencies (`pyproject.toml`)

**Files:**
- Modify: `pyproject.toml:76-82`

- [ ] **Step 1: Add email optional dependency group**

In `pyproject.toml`, after the `neural` group (line 81), add:
```toml
email = [
    "google-api-python-client>=2.130",
    "google-auth-oauthlib>=1.2",
    "google-auth-httplib2>=0.2",
]
```

And update the `all` group (line 82) to include `email`:
```toml
all = ["homie-ai[model,voice,context,storage,app,neural,email]"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat(email): add email optional dependency group to pyproject.toml"
```

---

### Task 11: Daemon Integration

**Files:**
- Modify: `src/homie_app/daemon.py`
- Modify: `src/homie_core/email/__init__.py`

- [ ] **Step 1: Create EmailService facade in `__init__.py`**

Update `src/homie_core/email/__init__.py`:
```python
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
        # Get OAuth client credentials for token refresh
        client_cred = self._vault.get_credential("gmail", account_id="oauth_client")
        client_id = client_cred.access_token if client_cred else ""
        client_secret = client_cred.refresh_token if client_cred else ""

        for cred in credentials:
            # Credential is a dataclass — use attribute access, not dict access
            if not cred.active:
                continue
            if cred.account_id == "oauth_client":
                continue  # Skip the OAuth client credential itself
            account_id = cred.account_id
            try:
                provider = GmailProvider(account_id=account_id)
                provider.authenticate(cred, vault=self._vault,
                                      client_id=client_id, client_secret=client_secret)
                classifier = EmailClassifier(user_email=account_id)

                # Ensure Homie labels exist and get their IDs
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

                # Write-through check_interval → connection_status.sync_interval (spec Section 10.3)
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
        """Create Homie/* labels if they don't exist. Returns category→label_id map."""
        from homie_core.email.organizer import HOMIE_LABELS
        existing = {l.name: l.id for l in provider.list_labels()}
        label_ids = {}
        for category, label_name in HOMIE_LABELS.items():
            if label_name in existing:
                label_ids[category] = existing[label_name]
            else:
                try:
                    # Create via Gmail API
                    result = provider._service.users().labels().create(
                        userId="me",
                        body={"name": label_name, "labelListVisibility": "labelShow",
                              "messageListVisibility": "show"},
                    ).execute()
                    label_ids[category] = result["id"]
                except Exception:
                    pass
        return label_ids
```

- [ ] **Step 2: Add email initialization to daemon.py**

In `src/homie_app/daemon.py`, after the vault sync manager init (~line 119), add:

```python
# Initialize email service if Gmail is connected
self._email_service = None
try:
    gmail_creds = self._vault.list_credentials("gmail")
    active_gmail = [c for c in gmail_creds if c.get("active", True)]
    if active_gmail:
        from homie_core.email import EmailService
        from homie_core.vault.schema import create_cache_db
        cache_path = storage / "cache.db"
        cache_conn = create_cache_db(cache_path)
        self._email_service = EmailService(
            vault=self._vault, cache_conn=cache_conn,
            working_memory=self._working_memory,
        )
        accounts = self._email_service.initialize()
        if accounts:
            self._vault_sync.register_callback("gmail", self._email_service.sync_tick)
            print(f"  Email: {len(accounts)} Gmail account(s) connected")
except Exception as e:
    print(f"  Email: not available ({e})")
```

In `_ensure_brain()`, after `register_builtin_tools(...)` (~line 288), add:

```python
# Register email tools if service is available
if self._email_service:
    from homie_core.email.tools import register_email_tools
    register_email_tools(tool_registry, self._email_service)
```

In `stop()`, before vault lock (~line 451), add:

```python
# Close email service
if self._email_service:
    self._email_service = None
```

- [ ] **Step 3: Run existing daemon tests to check no regressions**

Run: `pytest tests/unit/test_app/ -v`
Expected: All existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/homie_core/email/__init__.py src/homie_app/daemon.py
git commit -m "feat(email): add EmailService facade + daemon integration"
```

---

### Task 12: CLI Integration — `homie connect gmail` + `homie email`

**Files:**
- Modify: `src/homie_app/cli.py`

- [ ] **Step 1: Wire `homie connect gmail` to OAuth flow**

In `src/homie_app/cli.py`, update the `cmd_connect` function (currently a stub) to handle `gmail`:

```python
def cmd_connect(args, config):
    """Connect a provider via OAuth or API key."""
    provider = args.provider
    if provider != "gmail":
        print(f"Provider '{provider}' not yet supported. Available: gmail")
        return

    from homie_core.email.oauth import GmailOAuth, GMAIL_SCOPES
    from homie_core.vault.secure_vault import SecureVault
    from pathlib import Path
    import time

    storage = Path(config.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()

    # Check for existing OAuth client credentials (Credential is a dataclass — attribute access)
    client_cred = vault.get_credential("gmail", account_id="oauth_client")
    if client_cred:
        client_id = client_cred.access_token  # Client ID stored in access_token field
        client_secret = client_cred.refresh_token  # Client secret stored in refresh_token field
    else:
        print("\nGmail OAuth Setup")
        print("=" * 40)
        print("You need a Google Cloud OAuth client ID.")
        print("1. Go to https://console.cloud.google.com/apis/credentials")
        print("2. Create an OAuth 2.0 Client ID (Desktop app)")
        print("3. Enable the Gmail API\n")
        client_id = input("Client ID: ").strip()
        client_secret = input("Client Secret: ").strip()
        if not client_id or not client_secret:
            print("Cancelled.")
            vault.lock()
            return

        # Store client credentials
        vault.store_credential(
            provider="gmail", account_id="oauth_client",
            token_type="oauth_client",
            access_token=client_id,
            refresh_token=client_secret,
            scopes=GMAIL_SCOPES,
        )

    oauth = GmailOAuth(client_id=client_id, client_secret=client_secret)

    # Try local redirect first
    print("\nOpening browser for Google authorization...")
    auth_url = oauth.get_auth_url(use_local_server=True)

    import webbrowser
    webbrowser.open(auth_url)

    code = oauth.wait_for_redirect(timeout=120)

    if not code:
        # Fallback to manual
        print("\nLocal redirect failed. Manual authorization:")
        manual_url = oauth.get_auth_url(use_local_server=False)
        print(f"\nVisit this URL:\n{manual_url}\n")
        code = input("Paste the authorization code: ").strip()
        if not code:
            print("Cancelled.")
            vault.lock()
            return
        tokens = oauth.exchange(code, use_local_server=False)
    else:
        tokens = oauth.exchange(code, use_local_server=True)

    # Get profile to determine account email
    from types import SimpleNamespace
    from homie_core.email.gmail_provider import GmailProvider
    provider_instance = GmailProvider(account_id="pending")
    # Build a temporary credential-like object for initial auth (not yet stored in vault)
    temp_cred = SimpleNamespace(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        provider="gmail", account_id="pending",
    )
    provider_instance.authenticate(temp_cred, client_id=client_id, client_secret=client_secret)
    profile = provider_instance.get_profile()
    email_addr = profile["emailAddress"]

    # Store credentials
    vault.store_credential(
        provider="gmail", account_id=email_addr,
        token_type="oauth2",
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token", ""),
        scopes=GMAIL_SCOPES,
        expires_at=time.time() + tokens.get("expires_in", 3600),
    )

    vault.set_connection_status("gmail", connected=True, label=email_addr)
    vault.log_consent("gmail", "connected", scopes=GMAIL_SCOPES)

    print(f"\nConnected: {email_addr}")
    print("Run `homie email sync` to fetch recent emails, or they'll sync automatically.")
    vault.lock()
```

- [ ] **Step 2: Add `homie email` subcommands**

Add a new `email` subparser group and these commands:

```python
def cmd_email_summary(args, config):
    """Print email summary."""
    from homie_core.email import EmailService
    from homie_core.vault.secure_vault import SecureVault
    from homie_core.vault.schema import create_cache_db
    from pathlib import Path
    import json

    storage = Path(config.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()
    cache_conn = create_cache_db(storage / "cache.db")

    service = EmailService(vault=vault, cache_conn=cache_conn)
    accounts = service.initialize()
    if not accounts:
        print("No email accounts connected. Run: homie connect gmail")
        vault.lock()
        return

    summary = service.get_summary(days=int(getattr(args, "days", 1)))
    print(f"\nEmail Summary (last {getattr(args, 'days', 1)} day(s)):")
    print(f"  Total: {summary['total']}")
    print(f"  Unread: {summary['unread']}")
    if summary['high_priority']:
        print(f"\n  High Priority:")
        for item in summary['high_priority'][:5]:
            print(f"    - {item['sender']}: {item['subject']}")
    vault.lock()


def cmd_email_sync(args, config):
    """Force immediate email sync."""
    from homie_core.email import EmailService
    from homie_core.vault.secure_vault import SecureVault
    from homie_core.vault.schema import create_cache_db
    from pathlib import Path

    storage = Path(config.storage.path)
    vault = SecureVault(storage_dir=storage / "vault")
    vault.unlock()
    cache_conn = create_cache_db(storage / "cache.db")

    service = EmailService(vault=vault, cache_conn=cache_conn)
    accounts = service.initialize()
    if not accounts:
        print("No email accounts connected. Run: homie connect gmail")
        vault.lock()
        return

    print("Syncing...")
    result = service.sync_tick()
    print(result)
    vault.lock()


def cmd_email_config(args, config):
    """Show email sync configuration."""
    from homie_core.vault.schema import create_cache_db
    from pathlib import Path

    storage = Path(config.storage.path)
    cache_conn = create_cache_db(storage / "cache.db")

    rows = cache_conn.execute("SELECT * FROM email_config").fetchall()
    if not rows:
        print("No email configuration found. Defaults apply (5 min sync, high-priority notifications).")
        return

    for row in rows:
        print(f"\nAccount: {row[0]}")
        print(f"  Check interval: {row[1]}s")
        print(f"  Notify priority: {row[2]}")
        print(f"  Quiet hours: {row[3] or 'none'}-{row[4] or 'none'}")
        print(f"  Auto-trash spam: {'yes' if row[5] else 'no'}")
    cache_conn.close()
```

Add subparsers for `email` in the argparse setup:
```python
email_parser = subparsers.add_parser("email", help="Email management")
email_sub = email_parser.add_subparsers(dest="email_command")

email_summary_parser = email_sub.add_parser("summary", help="Email summary")
email_summary_parser.add_argument("--days", type=int, default=1, help="Days to summarize")

email_sub.add_parser("sync", help="Force sync now")
email_sub.add_parser("config", help="Show email settings")
```

Add dispatch:
```python
# In main() dispatch dict:
"email": lambda args, config: {
    "summary": cmd_email_summary,
    "sync": cmd_email_sync,
    "config": cmd_email_config,
}.get(args.email_command, lambda a, c: email_parser.print_help())(args, config),
```

- [ ] **Step 3: Add email meta-commands for chat mode**

In `_handle_meta_command`, add:
```python
elif cmd.startswith("/email"):
    parts = cmd.split()
    if len(parts) > 1 and parts[1] == "summary":
        return tool_email_summary(days=parts[2] if len(parts) > 2 else "1")
    elif len(parts) > 1 and parts[1] == "sync":
        return "Use the `homie email sync` CLI command for manual sync."
    else:
        return "Email commands: /email summary [days]"
```

- [ ] **Step 4: Run CLI tests to check no regressions**

Run: `pytest tests/unit/test_app/test_cli.py -v`
Expected: All existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/homie_app/cli.py
git commit -m "feat(email): wire CLI — connect gmail, email summary/sync/config commands"
```

---

### Task 12b: Update `__init__.py` exports and final integration test

**Files:**
- Modify: `src/homie_core/email/__init__.py` (add re-exports)

- [ ] **Step 1: Ensure `__init__.py` exports key classes**

The `__init__.py` already contains `EmailService`. Add re-exports at the bottom:

```python
from homie_core.email.gmail_provider import GmailProvider
from homie_core.email.models import (
    EmailMessage,
    EmailSyncConfig,
    SyncResult,
)

__all__ = ["EmailService", "GmailProvider", "EmailMessage", "EmailSyncConfig", "SyncResult"]
```

Note: These imports should be placed after the `EmailService` class definition, not at the top of the file, to avoid circular imports. The re-exports allow `from homie_core.email import GmailProvider`.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/unit/test_email/ -v`
Expected: All email tests pass (models, classifier, organizer, sync_engine, oauth, gmail_provider, tools).

Run: `pytest tests/ -v --timeout=60`
Expected: All tests pass (no regressions).

- [ ] **Step 3: Commit**

```bash
git add src/homie_core/email/__init__.py
git commit -m "feat(email): finalize module exports and verify full integration"
```
