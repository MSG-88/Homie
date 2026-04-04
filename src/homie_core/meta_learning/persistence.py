# src/homie_core/meta_learning/persistence.py
"""SQLite persistence layer for meta-learning state."""
from __future__ import annotations
import json, logging, sqlite3
from pathlib import Path
from typing import Any
from homie_core.utils import utc_now

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_strategy_records (task_type TEXT NOT NULL, strategy_key TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, successes INTEGER NOT NULL DEFAULT 0, total_duration_ms REAL NOT NULL DEFAULT 0.0, total_quality REAL NOT NULL DEFAULT 0.0, total_satisfaction REAL NOT NULL DEFAULT 0.0, last_used TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (task_type, strategy_key));
CREATE TABLE IF NOT EXISTS ml_strategy_registry (task_type TEXT NOT NULL, strategy_key TEXT NOT NULL, strategy_json TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (task_type, strategy_key));
CREATE TABLE IF NOT EXISTS ml_tuner_params (param_name TEXT PRIMARY KEY, value_json TEXT NOT NULL, prior_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ml_tuner_history (id INTEGER PRIMARY KEY AUTOINCREMENT, parameter TEXT NOT NULL, old_value TEXT NOT NULL, new_value TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', applied_at TEXT NOT NULL, reverted INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS ml_task_entries (id INTEGER PRIMARY KEY AUTOINCREMENT, task_type TEXT NOT NULL, strategy_key TEXT NOT NULL DEFAULT '', timestamp TEXT NOT NULL, duration_ms REAL NOT NULL, success INTEGER NOT NULL, quality_score REAL NOT NULL, satisfaction REAL NOT NULL DEFAULT 0.0, context_json TEXT NOT NULL DEFAULT '{}');
"""

class MetaLearningStore:
    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def initialize(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        if self._conn: self._conn.close(); self._conn = None

    @property
    def conn(self):
        if self._conn is None: raise RuntimeError("MetaLearningStore not initialised")
        return self._conn

    def upsert_strategy_record(self, task_type, strategy_key, attempts, successes, total_duration_ms, total_quality, total_satisfaction, last_used):
        now = utc_now().isoformat()
        self.conn.execute("INSERT INTO ml_strategy_records (task_type, strategy_key, attempts, successes, total_duration_ms, total_quality, total_satisfaction, last_used, updated_at) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(task_type, strategy_key) DO UPDATE SET attempts=excluded.attempts, successes=excluded.successes, total_duration_ms=excluded.total_duration_ms, total_quality=excluded.total_quality, total_satisfaction=excluded.total_satisfaction, last_used=excluded.last_used, updated_at=excluded.updated_at", (task_type, strategy_key, attempts, successes, total_duration_ms, total_quality, total_satisfaction, last_used, now))
        self.conn.commit()

    def load_strategy_records(self): return [dict(r) for r in self.conn.execute("SELECT * FROM ml_strategy_records").fetchall()]

    def upsert_strategy(self, task_type, strategy_key, strategy):
        now = utc_now().isoformat()
        self.conn.execute("INSERT INTO ml_strategy_registry (task_type, strategy_key, strategy_json, updated_at) VALUES (?,?,?,?) ON CONFLICT(task_type, strategy_key) DO UPDATE SET strategy_json=excluded.strategy_json, updated_at=excluded.updated_at", (task_type, strategy_key, json.dumps(strategy), now))
        self.conn.commit()

    def load_strategies(self):
        return [{**dict(r), "strategy": json.loads(r["strategy_json"])} for r in self.conn.execute("SELECT * FROM ml_strategy_registry").fetchall()]

    def upsert_tuner_param(self, param_name, value, prior=None):
        now = utc_now().isoformat()
        self.conn.execute("INSERT INTO ml_tuner_params (param_name, value_json, prior_json, updated_at) VALUES (?,?,?,?) ON CONFLICT(param_name) DO UPDATE SET value_json=excluded.value_json, prior_json=excluded.prior_json, updated_at=excluded.updated_at", (param_name, json.dumps(value), json.dumps(prior or {}), now))
        self.conn.commit()

    def load_tuner_params(self):
        return {r["param_name"]: {"value": json.loads(r["value_json"]), "prior": json.loads(r["prior_json"])} for r in self.conn.execute("SELECT * FROM ml_tuner_params").fetchall()}

    def add_tuner_history(self, parameter, old_value, new_value, reason):
        now = utc_now().isoformat()
        cur = self.conn.execute("INSERT INTO ml_tuner_history (parameter, old_value, new_value, reason, applied_at) VALUES (?,?,?,?,?)", (parameter, json.dumps(old_value), json.dumps(new_value), reason, now))
        self.conn.commit()
        return cur.lastrowid

    def mark_reverted(self, row_id):
        self.conn.execute("UPDATE ml_tuner_history SET reverted=1 WHERE id=?", (row_id,)); self.conn.commit()

    def load_tuner_history(self):
        return [{**dict(r), "old_value": json.loads(r["old_value"]), "new_value": json.loads(r["new_value"])} for r in self.conn.execute("SELECT * FROM ml_tuner_history ORDER BY applied_at DESC").fetchall()]

    def add_task_entry(self, task_type, strategy_key, duration_ms, success, quality_score, satisfaction=0.0, context=None):
        now = utc_now().isoformat()
        cur = self.conn.execute("INSERT INTO ml_task_entries (task_type, strategy_key, timestamp, duration_ms, success, quality_score, satisfaction, context_json) VALUES (?,?,?,?,?,?,?,?)", (task_type, strategy_key, now, duration_ms, int(success), quality_score, satisfaction, json.dumps(context or {})))
        self.conn.commit()
        return cur.lastrowid

    def load_task_entries(self, task_type=None, limit=500):
        if task_type: rows = self.conn.execute("SELECT * FROM ml_task_entries WHERE task_type=? ORDER BY timestamp DESC LIMIT ?", (task_type, limit)).fetchall()
        else: rows = self.conn.execute("SELECT * FROM ml_task_entries ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "success": bool(r["success"]), "context": json.loads(r["context_json"])} for r in rows]
