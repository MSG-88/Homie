"""Learning memory — SQLite tables for adaptive learning persistence."""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .observation.signals import LearningSignal


class LearningStorage:
    """SQLite-backed storage for all adaptive learning data."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Create database and all learning tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                signal_type TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                data TEXT NOT NULL,
                context TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_signals_cat ON learning_signals(category);
            CREATE INDEX IF NOT EXISTS idx_signals_ts ON learning_signals(timestamp);

            CREATE TABLE IF NOT EXISTS preference_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer_type TEXT NOT NULL,
                context_key TEXT NOT NULL,
                profile_data TEXT NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0.0,
                updated_at REAL NOT NULL,
                UNIQUE(layer_type, context_key)
            );

            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT NOT NULL,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                ttl REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_hit REAL
            );
            CREATE INDEX IF NOT EXISTS idx_cache_hash ON response_cache(query_hash);

            CREATE TABLE IF NOT EXISTS context_relevance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_type TEXT NOT NULL,
                context_source TEXT NOT NULL,
                relevance_score REAL NOT NULL DEFAULT 0.5,
                sample_count INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                UNIQUE(query_type, context_source)
            );

            CREATE TABLE IF NOT EXISTS resource_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(pattern_type, pattern_key)
            );

            CREATE TABLE IF NOT EXISTS project_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pk_subject ON project_knowledge(subject);

            CREATE TABLE IF NOT EXISTS behavioral_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                time_window TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(pattern_type, time_window)
            );

            CREATE TABLE IF NOT EXISTS decisions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision TEXT NOT NULL,
                domain TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_domain ON decisions_log(domain);

            CREATE TABLE IF NOT EXISTS customization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_text TEXT NOT NULL,
                generated_paths TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'active',
                version_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self._conn.commit()

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        if self._conn is None:
            return []
        cursor = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row["name"] for row in cursor.fetchall()]

    # --- Signal operations ---

    def write_signal(self, signal: LearningSignal) -> None:
        """Write a learning signal (append-only)."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO learning_signals (timestamp, signal_type, category, source, data, context, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (signal.timestamp, signal.signal_type.value, signal.category.value, signal.source, json.dumps(signal.data), json.dumps(signal.context), signal.confidence),
            )
            self._conn.commit()

    def query_signals(self, category: Optional[str] = None, source: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Query learning signals."""
        if self._conn is None:
            return []
        clauses, params = [], []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if source:
            clauses.append("source = ?")
            params.append(source)
        where = " AND ".join(clauses) if clauses else "1=1"
        cursor = self._conn.execute(
            f"SELECT * FROM learning_signals WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        )
        return [dict(row) for row in cursor.fetchall()]

    # --- Preference operations ---

    def save_preference(self, layer_type: str, context_key: str, profile_data: dict) -> None:
        """Save or update a preference profile."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                """INSERT INTO preference_profiles (layer_type, context_key, profile_data, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(layer_type, context_key) DO UPDATE SET
                   profile_data = excluded.profile_data, updated_at = excluded.updated_at""",
                (layer_type, context_key, json.dumps(profile_data), time.time()),
            )
            self._conn.commit()

    def get_preference(self, layer_type: str, context_key: str) -> Optional[dict]:
        """Get a preference profile."""
        if self._conn is None:
            return None
        cursor = self._conn.execute(
            "SELECT profile_data FROM preference_profiles WHERE layer_type = ? AND context_key = ?",
            (layer_type, context_key),
        )
        row = cursor.fetchone()
        return json.loads(row["profile_data"]) if row else None

    # --- Decision operations ---

    def write_decision(self, decision: str, domain: str, context: dict = None) -> None:
        """Log an extracted decision."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO decisions_log (decision, domain, context, created_at) VALUES (?, ?, ?, ?)",
                (decision, domain, json.dumps(context or {}), time.time()),
            )
            self._conn.commit()

    def query_decisions(self, domain: Optional[str] = None, limit: int = 50) -> list[dict]:
        """Query decisions log."""
        if self._conn is None:
            return []
        if domain:
            cursor = self._conn.execute(
                "SELECT * FROM decisions_log WHERE domain = ? ORDER BY created_at DESC LIMIT ?",
                (domain, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM decisions_log ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
