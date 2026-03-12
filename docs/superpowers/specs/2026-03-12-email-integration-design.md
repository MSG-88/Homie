# Email Integration — Design Spec

**Sub-project:** #1 (of 5)
**Depends on:** Sub-project #5 (Privacy & Secure Storage Layer) — completed
**Goal:** Enable Homie to connect to Gmail (and future email providers), read and analyze emails, organize the inbox, detect and remove spam, create draft replies, and proactively notify the user about important emails.

---

## 1. Architecture Overview

Email integration is a first-class core module (`src/homie_core/email/`) — not a plugin. The complexity (OAuth lifecycle, background sync, multi-account, spam classification, priority scoring, draft composition) warrants a dedicated module with its own provider abstraction.

**High-level data flow:**

```
User: "homie connect gmail"
    → OAuth flow (browser redirect or manual code)
    → Tokens stored encrypted in vault.db
    → Connection status set in cache.db
    → SyncManager callback registered

Daemon tick (every 60s):
    → SyncEngine.sync_incremental()
        → Gmail history API (delta since last sync)
        → Classifier scores each new message (spam + priority)
        → Organizer applies labels, archives low-priority
        → High-priority emails → ProactiveEngine notification
    → Results logged to daemon console

User: "what emails need my attention?"
    → Brain calls email_unread() tool
    → Returns priority-sorted list
    → Brain summarizes for user

User: "draft a reply to John about the deadline"
    → Brain calls email_search() → email_read() → email_draft()
    → Draft saved in Gmail (user reviews and sends manually)
```

**Key principle:** Homie never sends emails autonomously. All compose actions create drafts only.

---

## 2. Module Structure

```
src/homie_core/email/
├── __init__.py              # Re-exports EmailService, GmailProvider
├── provider.py              # Abstract EmailProvider interface
├── gmail_provider.py        # Gmail API implementation
├── oauth.py                 # OAuth 2.0 flow (local redirect + manual fallback)
├── sync_engine.py           # Incremental sync with historyId tracking
├── classifier.py            # Spam detection + priority scoring
├── organizer.py             # Auto-labeling, archiving, priority assignment
├── models.py                # EmailMessage, EmailThread, Label, SyncState, EmailSyncConfig
└── tools.py                 # Tool wrappers for ToolRegistry

src/homie_core/vault/schema.py   # (Modify) Add migration v2 with email tables
src/homie_app/cli.py             # (Modify) Wire `homie connect gmail` to OAuth
src/homie_app/daemon.py          # (Modify) Initialize EmailService, register sync
pyproject.toml                   # (Modify) Add [email] optional dependency
```

---

## 3. Provider Abstraction

```python
class EmailProvider(ABC):
    """Abstract email provider interface.

    Gmail implements this directly via google-api-python-client.
    Future providers (Outlook, IMAP) implement the same interface.
    """

    @abstractmethod
    def authenticate(self, credentials: Credential) -> None:
        """Authenticate with stored credentials. Refreshes token if expired."""

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
    def create_draft(self, to: str, subject: str, body: str,
                     reply_to: str | None = None) -> str:
        """Create a draft email. Returns draft ID."""

    @abstractmethod
    def list_labels(self) -> list[Label]:
        """List all labels/folders for the account."""

    @abstractmethod
    def get_profile(self) -> dict:
        """Get account profile (email address, display name)."""
```

**`HistoryChange` dataclass:**

```python
@dataclass
class HistoryChange:
    message_id: str
    change_type: str   # "added", "deleted", "labelAdded", "labelRemoved"
    labels: list[str] = field(default_factory=list)
```

---

## 4. Data Models

```python
@dataclass
class EmailMessage:
    id: str                          # Provider message ID
    thread_id: str                   # Thread/conversation ID
    account_id: str                  # user@gmail.com
    provider: str                    # "gmail"
    subject: str
    sender: str                      # "Name <email@example.com>"
    recipients: list[str]            # To + CC
    snippet: str                     # First ~200 chars preview
    body: str | None = None          # Full body (None if not yet fetched)
    labels: list[str] = field(default_factory=list)
    date: float = 0.0               # Unix timestamp
    is_read: bool = True
    is_starred: bool = False
    has_attachments: bool = False
    attachment_names: list[str] = field(default_factory=list)
    # Homie-assigned metadata
    priority: str = "medium"         # "high", "medium", "low"
    spam_score: float = 0.0          # 0.0 = clean, 1.0 = definite spam
    categories: list[str] = field(default_factory=list)  # "bill", "newsletter", etc.


@dataclass
class EmailThread:
    id: str
    account_id: str
    subject: str
    participants: list[str]
    message_count: int
    last_message_date: float
    snippet: str
    labels: list[str] = field(default_factory=list)


@dataclass
class SyncState:
    account_id: str
    provider: str
    history_id: str | None = None
    last_full_sync: float = 0.0
    last_incremental_sync: float = 0.0
    total_synced: int = 0


@dataclass
class Label:
    id: str
    name: str
    type: str = "user"               # "system" or "user"


@dataclass
class EmailSyncConfig:
    account_id: str
    check_interval: int = 300        # seconds between checks
    notify_priority: str = "high"    # minimum priority to notify: "high", "medium", "all", "none"
    quiet_hours_start: int | None = None  # hour (0-23), e.g., 22
    quiet_hours_end: int | None = None    # hour (0-23), e.g., 7
    auto_trash_spam: bool = True


@dataclass
class SyncResult:
    account_id: str
    new_messages: int = 0
    updated_messages: int = 0
    trashed_messages: int = 0
    notifications: list[EmailMessage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
```

---

## 5. OAuth Flow

### 5.1 Flow Overview

`homie connect gmail` triggers the OAuth 2.0 authorization code flow:

1. **Try local redirect server** — Start `http://localhost:8547/callback`, open browser to Google consent screen. On redirect, extract auth code from URL params, exchange for tokens.
2. **Fallback to manual** — If port unavailable or headless, print the authorization URL. User visits, approves, copies code from browser, pastes into terminal.
3. **Store credentials** — Access token, refresh token, expiry, and scopes encrypted in vault. Connection status and consent logged.

### 5.2 Google OAuth Scopes

```
gmail.readonly   — Read emails, labels, threads
gmail.modify     — Apply labels, trash, archive
gmail.compose    — Create drafts
```

### 5.3 Token Refresh

Before any Gmail API call, the provider checks `credential.expires_at`. If expired (or within 60s of expiry), it uses the refresh token to obtain a new access token and calls `vault.refresh_credential()`. This is thread-safe via the vault's per-credential locks.

If refresh fails (token revoked by user in Google settings):
- Mark connection as errored via `vault.set_connection_status(provider, connected=True, last_sync_error="Token revoked")`
- Log consent event: `vault.log_consent("gmail", "token_revoked")`
- Notify user: "Gmail connection lost. Run `homie connect gmail` to reconnect."

### 5.4 Multi-Account

Each `homie connect gmail` detects the authenticated email address from the profile API and stores credentials with `account_id=email_address`. Connecting again with a different Google account adds a second credential. All sync and tool operations iterate over `vault.list_credentials("gmail")`.

### 5.5 Client Credentials

The user provides their own Google Cloud OAuth client ID and secret during `homie connect gmail` (prompted interactively). Stored in vault as `gmail:oauth_client` credential. This avoids shipping embedded credentials in open-source code.

---

## 6. Sync Engine

### 6.1 Initial Sync

On first connection (or when `SyncState.history_id` is None):

1. Fetch messages from the last 7 days via `messages.list(q="newer_than:7d")`
2. Batch-fetch full message content using `BatchHttpRequest`
3. Store all messages in `cache.db` emails table
4. Run classifier + organizer on all messages
5. Record `history_id` from the response for future incremental syncs
6. Set `SyncState.last_full_sync = now`

### 6.2 Incremental Sync

On each daemon tick (when sync is due per `check_interval`):

1. Check token freshness, refresh if needed
2. Call `history.list(startHistoryId=last_id)` — returns only changes
3. For new messages: fetch content, store, classify, organize
4. For label changes: update local cache
5. For deletions: remove from local cache
6. High-priority new messages added to notification list
7. Update `SyncState.history_id` and `last_incremental_sync`
8. Return `SyncResult` with counts and notifications

### 6.3 Rate Limiting

Gmail API quota: 250 units/second/user. Costs:
- `messages.list`: 5 units (up to 500 IDs)
- `messages.get`: 5 units per message
- `history.list`: 2 units
- `BatchHttpRequest`: groups up to 100 gets in one HTTP call

Typical incremental tick (0-5 new emails): ~30 units. Well within limits.

On `HttpError 429`: skip this tick, retry next cycle. No exponential backoff needed at these volumes.

### 6.4 Error Handling

| Error | Action |
|-------|--------|
| `HttpError 429` (rate limit) | Skip tick, retry next cycle |
| `HttpError 401` (expired) | Refresh token. If refresh fails → mark errored, notify user |
| `HttpError 403` (scope revoked) | Mark disconnected, log consent, notify user |
| Network error | Skip tick, retry next cycle |
| Corrupt response | Log warning, skip message, continue |

---

## 7. Classifier

### 7.1 Spam Scoring

Weighted heuristic signals (no external ML dependency):

| Signal | Weight |
|--------|--------|
| Sender never received a reply from user | +0.3 |
| `Unsubscribe` header present + never opened by user | +0.2 |
| Bulk sender headers (`Precedence: bulk`, `List-Unsubscribe`) | +0.2 |
| Subject matches spam patterns (ALL CAPS, excessive punctuation) | +0.2 |
| Known spam phrases in subject/body | +0.1 |
| Sender in user's reply history (known contact) | -0.5 |
| Direct recipient (To field, not CC/BCC/list) | -0.2 |
| Sender domain matches user's domain | -0.3 |

**Score interpretation:**

| Range | Action |
|-------|--------|
| 0.0–0.3 | Clean — no action |
| 0.3–0.8 | Suspicious — label `Homie/Review`, keep in inbox |
| 0.8–1.0 | Spam — auto-trash if `auto_trash_spam` enabled |

### 7.2 Priority Scoring

| Priority | Criteria |
|----------|----------|
| **High** | Direct email from known contact + action keywords (deadline, urgent, payment, meeting, RSVP); reply in active thread user participated in |
| **Medium** | Direct email from unknown sender; newsletters user engages with; CC'd on work threads |
| **Low** | Mailing lists; automated notifications; marketing; social media alerts |

### 7.3 Category Detection

Labels assigned by content analysis:
- `Homie/Bills` — Invoices, payment confirmations, due date mentions, financial amounts
- `Homie/Work` — From contacts sharing user's work domain
- `Homie/Newsletters` — Detected mailing lists (List-Unsubscribe header + periodic sender)
- `Homie/Social` — Social media notification patterns (LinkedIn, Facebook, etc.)
- `Homie/Review` — Borderline spam for manual review

### 7.4 Learning from Corrections

When user un-trashes a message Homie trashed (detected on next sync via label change), or manually trashes something Homie left alone:

1. Store correction in `spam_corrections` table with original score and sender
2. Adjust per-sender weight: repeated corrections for same sender shift its base score
3. Over time, the classifier adapts to the user's preferences

---

## 8. Organizer

### 8.1 Auto-Labeling

After classification, the organizer applies Gmail labels via the API:
- Creates `Homie/Bills`, `Homie/Work`, `Homie/Newsletters`, `Homie/Social`, `Homie/Review` labels on first run
- Applies labels based on classifier categories
- Labels are additive — never removes existing user labels

### 8.2 Archive Rules

| Condition | Action |
|-----------|--------|
| Low priority + not direct recipient | Archive (remove INBOX label) |
| Newsletter + user hasn't opened last 3 from same sender | Archive |
| Social media notification | Archive |
| Spam score > 0.8 + `auto_trash_spam` enabled | Trash |
| Everything else | Keep in inbox |

### 8.3 Financial Extraction

When a message is categorized as `Homie/Bills`:
1. Extract amount (regex patterns for currency + number)
2. Extract due date (date parsing from body text)
3. Extract payee/description from subject
4. Store via `vault.store_financial(source=f"gmail:{msg_id}", category="bill", ...)`

This feeds into Sub-project #2 (Financial Management).

---

## 9. AI Tools

Seven tools registered with the ToolRegistry:

| Tool | Params | Description |
|------|--------|-------------|
| `email_search` | `query`, `account="all"`, `max_results="10"` | Search emails using Gmail query syntax |
| `email_read` | `message_id` | Fetch full body of a specific email |
| `email_thread` | `thread_id` | Get all messages in a conversation |
| `email_draft` | `to`, `subject`, `body`, `reply_to=None`, `account=None` | Create a draft (never sends) |
| `email_labels` | `account=None` | List all labels for an account |
| `email_summary` | `days="1"` | Summary of recent emails (count, highlights, action items) |
| `email_unread` | `account="all"` | List unread emails grouped by priority |

**Tool output format:** JSON strings truncated to 2000 chars, consistent with existing tool patterns.

**Example AI interaction:**
```
User: "What emails need my attention?"
Brain: email_unread(account="all")
→ Returns: [{"priority":"high","subject":"Payment due March 15","sender":"billing@util.com"}, ...]
Brain: "You have 3 high-priority emails: ..."
```

---

## 10. Notification System

### 10.1 When to Notify

A new email triggers a notification when:
- `priority == "high"` AND `spam_score < 0.3`
- User's `notify_priority` setting allows it
- Current time is outside quiet hours (if configured)

### 10.2 Notification Delivery

Uses Homie's existing `ProactiveEngine`:

```python
# In sync_engine after classifying a new message:
if should_notify(message, config):
    working_memory.update("email_alert", {
        "subject": message.subject,
        "sender": message.sender,
        "priority": message.priority,
        "snippet": message.snippet,
    })
```

The `ProactiveEngine` + `InterruptionModel` (already in the daemon) decide when to surface the notification via the overlay popup or daemon console, respecting the user's flow state.

### 10.3 Configurable Settings

Stored per-account in `email_config` table:

| Setting | Default | Description |
|---------|---------|-------------|
| `check_interval` | 300 (5 min) | Seconds between sync ticks |
| `notify_priority` | `"high"` | Minimum priority to notify: `"high"`, `"medium"`, `"all"`, `"none"` |
| `quiet_hours_start` | None | Hour (0-23) to start suppressing notifications |
| `quiet_hours_end` | None | Hour (0-23) to stop suppressing notifications |
| `auto_trash_spam` | True | Auto-trash emails with spam_score > 0.8 |

---

## 11. Database Schema (Migration v1 → v2)

New tables added to `cache.db` via schema migration:

```sql
CREATE TABLE IF NOT EXISTS emails (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    subject TEXT,
    sender TEXT,
    recipients TEXT,              -- JSON array
    snippet TEXT,
    body TEXT,
    labels TEXT,                  -- JSON array
    date REAL,
    is_read INTEGER DEFAULT 1,
    is_starred INTEGER DEFAULT 0,
    has_attachments INTEGER DEFAULT 0,
    attachment_names TEXT,         -- JSON array
    priority TEXT DEFAULT 'medium',
    spam_score REAL DEFAULT 0.0,
    categories TEXT,              -- JSON array
    fetched_at REAL
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
    corrected_action TEXT,        -- "not_spam" or "is_spam"
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

**Why `cache.db`?** Email content is non-sensitive metadata already stored on Google's servers. Plaintext storage avoids encryption overhead on search/read. OAuth tokens remain encrypted in `vault.db`. The email cache can be wiped and re-synced without data loss.

---

## 12. Dependencies

New optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
email = [
    "google-api-python-client>=2.130",
    "google-auth-oauthlib>=1.2",
    "google-auth-httplib2>=0.2",
]
all = ["homie-ai[model,voice,context,storage,app,neural,email]"]
```

---

## 13. CLI Integration

### 13.1 `homie connect gmail`

Currently a stub. Will be wired to `oauth.py`:

1. Prompt for Google Cloud client ID and secret (first time only)
2. Run OAuth flow (local redirect or manual)
3. Store tokens in vault
4. Run initial sync (7-day pull)
5. Print summary: "Connected! Synced N emails from user@gmail.com"

### 13.2 `homie disconnect gmail`

Already implemented with confirmation menu (deactivate vs delete). No changes needed — works with vault credential APIs.

### 13.3 New: `homie email` subcommand

```
homie email summary          — Print email summary (unread count, high-priority items)
homie email config            — Show/edit email sync settings
homie email sync              — Force immediate sync (bypasses interval timer)
```

---

## 14. Daemon Integration

In `daemon.py` `__init__`, after vault setup:

```python
# Initialize email service if Gmail is connected
self._email_service = None
try:
    gmail_cred = self._vault.get_credential("gmail")
    if gmail_cred:
        from homie_core.email import EmailService
        self._email_service = EmailService(vault=self._vault)
        self._email_service.initialize()
        self._vault_sync.register_callback("gmail", self._email_service.sync_tick)
        print("  Email: Gmail connected")
except Exception as e:
    print(f"  Email: not available ({e})")
```

Tool registration happens in `_ensure_brain()` when the brain is first loaded — email tools are registered alongside existing tools.

In `stop()`, the email service cleans up any open connections.

---

## 15. Security Considerations

| Concern | Mitigation |
|---------|------------|
| OAuth tokens at rest | AES-256-GCM encrypted in vault.db |
| Token in memory | Thread-safe access, zeroed on vault lock |
| Token refresh race | Per-credential locks in vault |
| Scope creep | Request minimal scopes (readonly + modify + compose) |
| Accidental send | Draft-only by design, no send capability |
| Email body exposure | Stored in cache.db (plaintext) — acceptable since data exists on Google's servers. Can be wiped with `homie email clear-cache` |
| Client credentials | User provides their own Google Cloud project credentials |
| Consent audit | Every connect/disconnect/revoke logged in vault consent_log |

---

## 16. Testing Strategy

| Component | Test approach |
|-----------|--------------|
| `models.py` | Unit tests for dataclass creation and serialization |
| `oauth.py` | Mock HTTP server, mock Google auth responses |
| `gmail_provider.py` | Mock `googleapiclient` service object, test all provider methods |
| `sync_engine.py` | Mock provider, test initial sync, incremental sync, error handling |
| `classifier.py` | Unit tests with sample emails, verify scoring thresholds |
| `organizer.py` | Mock provider + classifier, verify label application and archive rules |
| `tools.py` | Mock EmailService, verify tool registration and output format |
| `schema.py` migration | Test v1→v2 migration creates tables correctly |

All Gmail API calls are mocked — no real Google account needed for tests.

---

## 17. Future Extensions (Not in Scope)

- IMAP/SMTP provider for Outlook, Yahoo (via EmailProvider interface)
- Full send capability with confirmation (upgrade from draft-only)
- Attachment download and indexing
- Email-based task extraction into task graph
- Smart reply suggestions using LLM
- Email search via RAG pipeline (vector embeddings of email content)
