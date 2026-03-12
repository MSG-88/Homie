"""Database schema creation, integrity checks, and migration support.

Two databases:
- vault.db: encrypted fields (credentials, profiles, consent, financial)
- cache.db: plaintext caches (folder watches, content index, connection status)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

CURRENT_SCHEMA_VERSION = 1

_VAULT_DDL = """
CREATE TABLE IF NOT EXISTS credentials (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    account_id TEXT NOT NULL,
    token_type TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at REAL,
    scopes TEXT,
    active INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
    id TEXT PRIMARY KEY,
    display_name TEXT,
    email TEXT,
    phone TEXT,
    metadata TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS consent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    action TEXT NOT NULL,
    scopes TEXT,
    reason TEXT,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS financial_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    amount TEXT,
    currency TEXT,
    due_date REAL,
    status TEXT DEFAULT 'pending',
    reminded_at REAL,
    raw_extract TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS folder_watches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    label TEXT,
    scan_interval INTEGER DEFAULT 300,
    last_scanned REAL,
    file_count INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS content_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    content_type TEXT NOT NULL,
    summary TEXT,
    topics TEXT,
    embeddings BLOB,
    indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS connection_status (
    provider TEXT PRIMARY KEY,
    connected INTEGER DEFAULT 0,
    display_label TEXT,
    connection_mode TEXT DEFAULT 'always_on',
    sync_interval INTEGER DEFAULT 300,
    last_sync REAL,
    last_sync_error TEXT
);

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
"""

_MIGRATIONS: dict[int, object] = {}


def create_vault_db(path: Path) -> sqlite3.Connection:
    """Create vault.db with WAL journal mode and all tables."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_VAULT_DDL)
    conn.commit()
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return conn


def create_cache_db(path: Path) -> sqlite3.Connection:
    """Create cache.db with WAL journal mode and all tables."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_CACHE_DDL)
    conn.commit()
    return conn


def get_schema_version(meta_path: Path) -> int:
    """Read the current schema version from vault.meta.json."""
    if not meta_path.exists():
        return 0
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("schema_version", 0)
    except Exception:
        return 0


def _set_schema_version(meta_path: Path, version: int) -> None:
    """Write the schema version to vault.meta.json, preserving other keys."""
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    meta["schema_version"] = version
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        meta_path.chmod(0o600)
    except OSError:
        pass


def run_migrations(conn: sqlite3.Connection, meta_path: Path) -> None:
    """Run any pending schema migrations in order."""
    current = get_schema_version(meta_path)
    for version in sorted(_MIGRATIONS.keys()):
        if version > current:
            _MIGRATIONS[version](conn)
            conn.commit()
    _set_schema_version(meta_path, CURRENT_SCHEMA_VERSION)


def check_integrity(conn: sqlite3.Connection) -> bool:
    """Run SQLite integrity check. Returns True if database is healthy."""
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        return result is not None and result[0] == "ok"
    except Exception:
        return False
