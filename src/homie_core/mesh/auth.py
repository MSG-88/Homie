"""Mesh authentication and role-based access control."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional
from homie_core.mesh.events import generate_ulid
from homie_core.utils import utc_now


class Role(IntEnum):
    VIEWER = 0    # Read-only dashboard access
    USER = 1      # Use Homie + configure own node
    ADMIN = 2     # Full control: manage nodes, users, settings


@dataclass
class MeshUser:
    user_id: str
    username: str
    role: Role
    node_id: str  # Home node
    created_at: str = field(default_factory=lambda: utc_now().isoformat())
    api_key_hash: str = ""
    active: bool = True

    def can_execute_tasks(self) -> bool:
        return self.role >= Role.USER

    def can_manage_nodes(self) -> bool:
        return self.role >= Role.ADMIN

    def can_manage_users(self) -> bool:
        return self.role >= Role.ADMIN

    def can_view_dashboard(self) -> bool:
        return self.role >= Role.VIEWER

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id, "username": self.username,
            "role": self.role.name, "node_id": self.node_id,
            "created_at": self.created_at, "active": self.active,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MeshUser:
        role_map = {"VIEWER": Role.VIEWER, "USER": Role.USER, "ADMIN": Role.ADMIN}
        return cls(
            user_id=d["user_id"], username=d["username"],
            role=role_map.get(d.get("role", "USER"), Role.USER),
            node_id=d.get("node_id", ""), created_at=d.get("created_at", ""),
            active=d.get("active", True),
        )


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    import secrets
    return f"homie_{secrets.token_urlsafe(32)}"


class AuthStore:
    """SQLite store for mesh users and API keys."""

    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS mesh_users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL DEFAULT 'USER',
                node_id TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                api_key_hash TEXT DEFAULT '',
                active BOOLEAN DEFAULT 1
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT DEFAULT '',
                details_json TEXT DEFAULT '{}',
                source_node TEXT DEFAULT '',
                target_node TEXT DEFAULT ''
            )
        """)
        self._conn.commit()

    def create_user(self, username: str, role: Role, node_id: str = "") -> tuple[MeshUser, str]:
        """Create a user and return (user, api_key)."""
        api_key = generate_api_key()
        user = MeshUser(
            user_id=generate_ulid(), username=username, role=role,
            node_id=node_id, api_key_hash=hash_api_key(api_key),
        )
        self._conn.execute("""
            INSERT INTO mesh_users (user_id, username, role, node_id, created_at, api_key_hash, active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user.user_id, user.username, user.role.name, user.node_id,
              user.created_at, user.api_key_hash, 1))
        self._conn.commit()
        return user, api_key

    def authenticate(self, api_key: str) -> Optional[MeshUser]:
        """Authenticate by API key. Returns user or None."""
        key_hash = hash_api_key(api_key)
        row = self._conn.execute(
            "SELECT * FROM mesh_users WHERE api_key_hash = ? AND active = 1",
            (key_hash,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row)

    def get_user(self, user_id: str) -> Optional[MeshUser]:
        row = self._conn.execute(
            "SELECT * FROM mesh_users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_name(self, username: str) -> Optional[MeshUser]:
        row = self._conn.execute(
            "SELECT * FROM mesh_users WHERE username = ?", (username,)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self) -> list[MeshUser]:
        rows = self._conn.execute("SELECT * FROM mesh_users ORDER BY created_at").fetchall()
        return [self._row_to_user(r) for r in rows]

    def deactivate_user(self, user_id: str) -> None:
        self._conn.execute("UPDATE mesh_users SET active = 0 WHERE user_id = ?", (user_id,))
        self._conn.commit()

    def log_action(self, user_id: str, action: str, target: str = "",
                   details: dict = None, source_node: str = "", target_node: str = "") -> None:
        self._conn.execute("""
            INSERT INTO audit_log (timestamp, user_id, action, target, details_json, source_node, target_node)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (utc_now().isoformat(), user_id, action, target,
              json.dumps(details or {}), source_node, target_node))
        self._conn.commit()

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> MeshUser:
        role_map = {"VIEWER": Role.VIEWER, "USER": Role.USER, "ADMIN": Role.ADMIN}
        return MeshUser(
            user_id=row["user_id"], username=row["username"],
            role=role_map.get(row["role"], Role.USER),
            node_id=row["node_id"], created_at=row["created_at"],
            api_key_hash=row["api_key_hash"], active=bool(row["active"]),
        )
