# Privacy & Secure Storage Layer — Design Spec

**Date:** 2026-03-12
**Sub-project:** #5 of 5 (Foundation layer)
**Branch:** feat/homie-ai-v2
**Dependencies:** None (other sub-projects depend on this)

---

## Overview

A local-first, encrypted storage layer that manages credentials, user profiles, consent records, and financial data for Homie's integrations (email, social media, local folders). All sensitive data stays on the user's machine, encrypted at rest with AES-256-GCM, with the master key protected by the OS keyring.

This module is the foundation for all integration sub-projects:
- Sub-project 1: Email Integration
- Sub-project 2: Financial Intelligence
- Sub-project 3: Local Folder Awareness
- Sub-project 4: Social & Communication Connectors

---

## Encryption Architecture

### Key Hierarchy

```
OS Keyring (Windows Credential Manager / macOS Keychain / Linux Secret Service)
  └── master_key (32 bytes, os.urandom)
      └── Optional password layer: PBKDF2(password, salt, 600k iterations)
          └── HKDF-SHA256(master_key, info=category_name) per category:
              ├── credentials_key
              ├── profiles_key
              ├── financial_key
              └── consent_key
```

### Encryption Details

- **Algorithm:** AES-256-GCM (authenticated encryption with tamper detection)
- **Per-field encryption:** Each sensitive field encrypted independently with its own random 12-byte nonce
- **Storage format:** `nonce (12B) || ciphertext || tag (16B)`, base64-encoded in SQLite TEXT columns
- **Key derivation:** `HKDF(algorithm=SHA256, length=32, salt=None, info=b"credentials")` — the category name is passed as the `info` parameter (context), not the `salt`. This follows RFC 5869 correctly: salt is optional randomness, info is application-specific context.
- **Password layer:** Optional. When enabled, the master key is encrypted with a password-derived key before storage in the OS keyring:
  1. User provides password
  2. `password_key = PBKDF2(password, salt, 600k iterations)` → 32-byte key
  3. `encrypted_master = AES-256-GCM(password_key, master_key)` → stored in keyring
  4. `salt` stored alongside in keyring (service: `homie-vault`, username: `password-salt`)
  5. On unlock: retrieve encrypted master + salt → PBKDF2 derive → AES-GCM decrypt → master key
- **Key material:** Stored in mutable `bytearray` (not `bytes`) so keys can be zeroed on lock

---

## Database Schema

Two SQLite databases with clear separation of encrypted and plaintext data.

### vault.db (encrypted fields, file permissions 0600)

```sql
-- OAuth tokens, API keys, refresh tokens
CREATE TABLE credentials (
    id TEXT PRIMARY KEY,           -- "gmail:user@example.com" (provider:account_id)
    provider TEXT NOT NULL,        -- "gmail", "linkedin", "slack", etc.
    account_id TEXT NOT NULL,      -- User identifier (email, username, etc.)
    token_type TEXT NOT NULL,      -- "oauth2", "api_key", "app_password"
    access_token TEXT NOT NULL,    -- AES-256-GCM encrypted
    refresh_token TEXT,            -- AES-256-GCM encrypted
    expires_at REAL,               -- Unix timestamp (plaintext for query efficiency)
    scopes TEXT,                   -- JSON array of granted scopes (plaintext)
    active INTEGER DEFAULT 1,     -- 1 = active, 0 = deactivated (credentials kept encrypted)
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
-- Supports multiple accounts per provider (e.g., two Gmail accounts)

-- User identity and linked account metadata
CREATE TABLE user_profiles (
    id TEXT PRIMARY KEY,           -- "primary" or provider-specific
    display_name TEXT,             -- AES-256-GCM encrypted
    email TEXT,                    -- AES-256-GCM encrypted
    phone TEXT,                    -- AES-256-GCM encrypted
    metadata TEXT,                 -- AES-256-GCM encrypted JSON
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Full audit trail of user consent decisions (append-only)
-- Stored in vault.db for integrity; reason field encrypted as it may contain user text
CREATE TABLE consent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,        -- "gmail", "linkedin", etc.
    action TEXT NOT NULL,          -- "connected", "declined", "disconnected", "reauthorized"
    scopes TEXT,                   -- What was granted/revoked (plaintext, not sensitive)
    reason TEXT,                   -- AES-256-GCM encrypted (may contain user-authored text)
    timestamp REAL NOT NULL
);

-- Bills, payments, due dates extracted from emails
CREATE TABLE financial_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- "gmail:msg_id_123"
    category TEXT NOT NULL,        -- "bill", "payment", "subscription", "reminder"
    description TEXT NOT NULL,     -- AES-256-GCM encrypted
    amount TEXT,                   -- AES-256-GCM encrypted (string to avoid float issues)
    currency TEXT,                 -- "USD", "INR", etc. (plaintext)
    due_date REAL,                 -- Unix timestamp (plaintext for query efficiency)
    status TEXT DEFAULT 'pending', -- "pending", "paid", "overdue", "dismissed"
    reminded_at REAL,              -- Last reminder timestamp
    raw_extract TEXT,              -- AES-256-GCM encrypted (original email snippet)
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
```

### cache.db (plaintext, file permissions 0644)

```sql
-- Folder watch configuration
CREATE TABLE folder_watches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    label TEXT,
    scan_interval INTEGER DEFAULT 300,
    last_scanned REAL,
    file_count INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1
);

-- Content analysis results and indexes
-- NOTE: summaries derived from encrypted sources (emails) are sanitized via
-- redact_sensitive_text() before storage. They contain topic abstractions,
-- not raw email content. If stronger guarantees are needed, summaries from
-- email sources can be moved to vault.db in a future iteration.
CREATE TABLE content_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- "folder:/path/file" or "email:msg_id"
    content_type TEXT NOT NULL,    -- "document", "code", "image", "spreadsheet"
    summary TEXT,                  -- Sanitized summary (no raw sensitive data)
    topics TEXT,                   -- JSON array of detected topics
    embeddings BLOB,              -- Vector embeddings for RAG
    indexed_at REAL NOT NULL
);

-- Connection status (fast lookup without decrypting vault)
CREATE TABLE connection_status (
    provider TEXT PRIMARY KEY,
    connected INTEGER DEFAULT 0,
    display_label TEXT,            -- "Gmail (m***@gmail.com)" - partially masked
    connection_mode TEXT DEFAULT 'always_on',  -- "always_on" or "on_demand"
    sync_interval INTEGER DEFAULT 300,         -- seconds, per provider
    last_sync REAL,
    last_sync_error TEXT
);
```

---

## SecureVault API

### Thread Safety

`SecureVault` is thread-safe. All database operations use a `threading.Lock` per database connection. SQLite is opened with `WAL` journal mode for concurrent read performance. The daemon's multiple threads (main, observer, scheduler) can safely call vault methods concurrently.

### Exception Hierarchy

```python
class VaultError(Exception): ...              # Base for all vault errors
class VaultLockedError(VaultError): ...       # Operation attempted while vault is locked
class VaultAuthError(VaultError): ...         # Wrong password or corrupted master key
class VaultCorruptError(VaultError): ...      # DB corruption or tampered ciphertext (GCM auth fail)
class CredentialNotFoundError(VaultError): ...# Requested credential does not exist
class RateLimitError(VaultError): ...         # Too many failed unlock attempts
```

### API Surface

```python
class SecureVault:
    """Single entry point for all secure storage operations.

    Thread-safe. All methods raise VaultLockedError if called before unlock().
    """

    def __init__(self, storage_dir: str | Path = "~/.homie/vault"): ...

    # Lifecycle
    def unlock(self, password: str | None = None) -> None:
        """Unlock vault. Raises VaultAuthError on wrong password,
        RateLimitError after 5 consecutive failures (60s cooldown)."""
    def lock(self) -> None:                     # Zeros all derived keys in memory
    def is_unlocked(self) -> bool: ...
    def set_password(self, password: str) -> None: ...
    def export_vault(self, path: Path, password: str) -> None:
        """Export as AES-256-GCM encrypted file. Format: JSON with all tables,
        encrypted with PBKDF2-derived key from the provided password."""
    def import_vault(self, path: Path, password: str) -> None: ...

    # Credentials (supports multiple accounts per provider via credential_id)
    def store_credential(self, provider: str, account_id: str,
                        token_type: str, access_token: str,
                        refresh_token: str | None = None,
                        expires_at: float | None = None,
                        scopes: list[str] | None = None) -> str:
        """Returns credential_id ('{provider}:{account_id}')."""
    def get_credential(self, provider: str,
                      account_id: str | None = None) -> Credential | None:
        """If account_id is None, returns first active credential for provider."""
    def list_credentials(self, provider: str) -> list[Credential]:
        """List all credentials (active + inactive) for a provider."""
    def refresh_credential(self, credential_id: str,
                          new_access_token: str,
                          new_expires_at: float | None = None) -> None:
        """Atomic token refresh. Uses a per-credential lock to prevent
        race conditions when SyncManager and user queries both detect expiry."""
    def deactivate_credential(self, credential_id: str) -> None:
        """Sets active=0. Credentials stay encrypted in vault. Reversible."""
    def delete_credential(self, credential_id: str) -> None:
        """Permanent removal. Caller must confirm with user first.
        Runs VACUUM on vault.db to reclaim space from deleted rows."""

    # User Profiles
    def store_profile(self, profile_id: str, display_name: str | None = None,
                     email: str | None = None, phone: str | None = None,
                     metadata: dict | None = None) -> None:
        """profile_id: 'primary' or '{provider}:{account_id}'."""
    def get_profile(self, profile_id: str) -> UserProfile | None: ...

    # Consent Management
    def log_consent(self, provider: str, action: str,
                   scopes: list[str] | None = None,
                   reason: str | None = None) -> None: ...
    def get_consent_history(self, provider: str) -> list[ConsentEntry]: ...
    def get_last_consent(self, provider: str) -> ConsentEntry | None: ...

    # Financial Data
    def store_financial(self, source: str, category: str,
                       description: str, amount: str | None = None,
                       currency: str | None = None,
                       due_date: float | None = None) -> int: ...
    def query_financial(self, status: str | None = None,
                       due_before: float | None = None,
                       category: str | None = None) -> list[FinancialRecord]: ...
    def update_financial(self, record_id: int, **kwargs) -> None: ...

    # Connection Status (reads from cache.db, no decryption needed)
    def set_connection_status(self, provider: str, connected: bool,
                             label: str | None = None,
                             mode: str = "always_on",
                             sync_interval: int = 300) -> None: ...
    def get_connection_status(self, provider: str) -> ConnectionStatus | None: ...
    def get_all_connections(self) -> list[ConnectionStatus]: ...
```

---

## Consent & Connection Flow

### First-time connection (just-in-time prompt)

When a feature needs a provider that isn't connected:

1. Homie explains what access it needs and why, in plain language
2. User approves or declines
3. If approved: OAuth2 browser flow, credentials stored, consent logged
4. User is asked about connection mode:
   - **Always connected (default):** Periodic background sync at configurable intervals
   - **Auto-disconnect (on_demand):** Connect only when needed, disconnect after sync
5. If declined: consent logged as "declined", Homie respects it with a 7-day re-ask cooldown

### Default behavior: always connected

When `connection_mode = "always_on"`:
- Background sync at provider-specific intervals (email: 5min, social: 30min, finance: 6hr)
- Token auto-refresh on expiry (silent, notify on failure)
- Data retrieved and stored locally continuously

When `connection_mode = "on_demand"`:
- Connect only when Homie needs data or user asks
- Disconnect after sync completes
- No background polling

### Disconnection flow

User-initiated disconnect presents three options:
1. **Disconnect (keep credentials encrypted)** — can reconnect later without re-auth
2. **Disconnect and delete credentials permanently** — full removal after confirmation
3. **Cancel**

Credentials are never deleted without explicit user confirmation.

### CLI commands

```
homie connect [provider]              # OAuth flow + consent
homie connect [provider] --mode X     # Set always_on or on_demand
homie connect [provider] --sync-interval N  # Seconds between syncs
homie disconnect [provider]           # Deactivate with confirmation dialog
homie connections                     # List all connection statuses
homie consent-log [provider]          # Full audit trail
```

---

## Master Key Lifecycle

### First run
1. Generate 32-byte random key via `os.urandom(32)`
2. Store in OS keyring (service: `homie-vault`, username: `master-key`)
3. Create `vault.db` and `cache.db` with schema
4. Derive category keys via HKDF
5. Optionally ask user to set a master password

### Session start (daemon boot)
1. Retrieve master key from OS keyring
2. If password-protected: prompt for password, PBKDF2 derive, decrypt
3. Derive category keys via HKDF, hold in memory as `bytearray`
4. Start SyncManager for always-on providers

### Session end / lock
1. Zero all derived keys in memory (overwrite `bytearray` with `\x00`)
2. Close DB connections
3. Read `connection_status` table for providers with `connection_mode = 'on_demand'` and revoke their active tokens

---

## Edge Cases & Resilience

### Keyring unavailable
On headless Linux (no D-Bus Secret Service) or CI environments:
1. `keyring` library is tried first
2. If it fails, fall back to file-based key storage at `~/.homie/vault/.keyfile` (permissions 0600)
3. Print a warning: "OS keyring not available. Using file-based key storage — set a master password for security."
4. If file-based fallback is used, strongly recommend (but don't force) setting a master password

### Corrupt database recovery
- SQLite is opened with `PRAGMA journal_mode=WAL` for crash safety
- On startup, run `PRAGMA integrity_check` on both databases
- If corruption detected: attempt `PRAGMA recover` (SQLite 3.29+), log the event
- If recovery fails: rename corrupt file to `vault.db.corrupt.{timestamp}`, create fresh database, warn user

### Concurrent daemon instances
- On startup, create a PID lock file at `~/.homie/vault/.lock`
- If lock file exists and the PID is still running, refuse to start with a clear error
- Stale lock files (PID not running) are cleaned up automatically

### Token refresh race condition
- `refresh_credential()` uses a per-credential lock (keyed by credential_id)
- If SyncManager and a user query both detect expiry, only the first caller performs the refresh
- The second caller re-reads the credential and finds a fresh token

### Master key rotation (future consideration)
- Not in initial implementation scope
- When added: generate new master key, re-derive all category keys, re-encrypt all fields, atomic swap via SQLite transaction

### Schema migrations
- `vault.meta.json` tracks `schema_version` (integer, starts at 1)
- `schema.py` contains a migration registry: `{version: migration_fn}`
- On startup, compare `schema_version` to latest, run pending migrations in order within a transaction
- Migrations are forward-only (no rollback)

### Brute force rate limiting persistence
- Failed unlock attempt count and lockout timestamp stored in `vault.meta.json`
- Survives daemon restarts — attacker cannot reset counter by restarting the process
- Counter resets only after a successful unlock

---

## Security Hardening

| Threat | Mitigation |
|--------|-----------|
| Memory dump | Zero keys on lock; keys in mutable `bytearray` not `bytes` |
| DB file stolen | Per-field AES-256-GCM; useless without master key from OS keyring |
| Keyring compromised | Optional password layer keeps master key encrypted |
| Token leak in logs | Existing `redact_sensitive_text()` catches OAuth token patterns |
| Prompt injection via email | Existing `injection_detector.py` scans external content |
| Brute force password | PBKDF2 600k iterations; rate limit: 5 failures → 60s cooldown |
| Stale tokens | Auto-refresh with failure notification; never store plaintext passwords |
| Backup exposure | `export_vault()` always password-encrypted (AES-256-GCM with PBKDF2-derived key), independent of keyring |
| Timing metadata exposure | `expires_at` and `due_date` stored plaintext for query efficiency — reveals token issuance timing. Accepted trade-off: this metadata is low-sensitivity compared to the tokens themselves |
| Deleted data remnants | `delete_credential()` runs `VACUUM` to reclaim free pages. On SSDs with wear leveling, true secure deletion is impossible — acknowledged limitation. Password layer provides defense-in-depth |

### File permissions

```
~/.homie/vault/
├── vault.db          # 0600 (owner read/write only)
├── cache.db          # 0644 (readable, no secrets)
└── vault.meta.json   # 0600 (schema version, created_at)
```

---

## Module Structure

```
src/homie_core/vault/
├── __init__.py
├── secure_vault.py       # SecureVault class — main API
├── encryption.py         # AES-256-GCM, HKDF, PBKDF2 key derivation
├── keyring_backend.py    # OS keyring read/write, password layer
├── schema.py             # DB table creation, migrations
├── models.py             # Credential, UserProfile, ConsentEntry,
│                         #   FinancialRecord, ConnectionStatus dataclasses
└── sync_manager.py       # Periodic background sync (always_on / on_demand)
```

### Integration with Homie

```python
# In HomieDaemon.__init__():
self._vault = SecureVault(storage_dir=storage / "vault")
self._vault.unlock()

# In HomieDaemon.start():
self._sync_manager = SyncManager(vault=self._vault)
self._sync_manager.start()

# In HomieDaemon.stop():
self._sync_manager.stop()
self._vault.lock()
```

SyncManager reuses the daemon's existing 60-second tick pattern rather than spawning a new thread.

### Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| `cryptography` | AES-256-GCM, HKDF, PBKDF2 | Well-maintained, widely used |
| `keyring` | OS keyring abstraction | ~50KB, pure Python + platform backends |

SQLite is in Python's stdlib. No other new dependencies.

---

## Build Order (Full Project)

This module (#5) is built first. Subsequent sub-projects in order:

1. **Email Integration (#1)** — Gmail/IMAP, uses vault for OAuth tokens
2. **Financial Intelligence (#2)** — builds on email data, uses vault for financial records
3. **Local Folder Awareness (#3)** — uses cache.db for folder config and indexes
4. **Social & Communication Connectors (#4)** — uses vault for social OAuth tokens and consent
