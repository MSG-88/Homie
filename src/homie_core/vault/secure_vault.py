"""SecureVault — thread-safe encrypted storage API."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from homie_core.vault.encryption import (
    generate_master_key, derive_category_key,
    encrypt_field, decrypt_field, zero_bytearray,
)
from homie_core.vault.exceptions import VaultLockedError, VaultAuthError
from homie_core.vault.keyring_backend import KeyringBackend
from homie_core.vault.models import (
    Credential, UserProfile, ConsentEntry, FinancialRecord, ConnectionStatus,
)
from homie_core.vault.schema import (
    create_vault_db, create_cache_db, check_integrity, run_migrations,
)


def _require_unlocked(method):
    """Decorator that raises VaultLockedError if vault is locked."""
    def wrapper(self, *args, **kwargs):
        if not self._unlocked:
            raise VaultLockedError("Vault is locked. Call unlock() first.")
        return method(self, *args, **kwargs)
    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


class SecureVault:
    """Thread-safe encrypted local vault for credentials and user data."""

    def __init__(self, storage_dir: str | Path = "~/.homie/vault"):
        self._storage_dir = Path(storage_dir).expanduser()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._keyring = KeyringBackend(fallback_dir=self._storage_dir)
        self._vault_conn = None
        self._cache_conn = None
        self._vault_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._credential_locks: dict[str, threading.Lock] = {}
        self._credential_locks_lock = threading.Lock()
        self._keys: dict[str, bytearray] = {}
        self._unlocked = False

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked

    @property
    def has_password(self) -> bool:
        return self._keyring.has_password()

    def unlock(self, password: Optional[str] = None) -> None:
        master = self._keyring.retrieve_master_key(password=password)
        if master is None:
            master = generate_master_key()
            self._keyring.store_master_key(master)
        for category in ("credentials", "profiles", "financial", "consent"):
            self._keys[category] = derive_category_key(master, category)
        zero_bytearray(master)
        vault_path = self._storage_dir / "vault.db"
        cache_path = self._storage_dir / "cache.db"
        self._vault_conn = create_vault_db(vault_path)
        self._cache_conn = create_cache_db(cache_path)
        meta_path = self._storage_dir / "vault.meta.json"
        run_migrations(self._vault_conn, meta_path)
        check_integrity(self._vault_conn)
        check_integrity(self._cache_conn)
        self._unlocked = True

    def lock(self) -> None:
        for key in self._keys.values():
            zero_bytearray(key)
        self._keys.clear()
        if self._vault_conn:
            self._vault_conn.close()
            self._vault_conn = None
        if self._cache_conn:
            self._cache_conn.close()
            self._cache_conn = None
        self._unlocked = False

    def set_password(self, password: str) -> None:
        if not self._unlocked:
            raise VaultLockedError("Vault must be unlocked to set password.")
        master = generate_master_key()
        new_keys = {}
        for category in ("credentials", "profiles", "financial", "consent"):
            new_keys[category] = derive_category_key(master, category)
        self._reencrypt_all(self._keys, new_keys)
        self._keyring.store_master_key(master)
        self._keyring.set_password(password, master)
        for old_key in self._keys.values():
            zero_bytearray(old_key)
        self._keys = new_keys
        zero_bytearray(master)

    # ── Export / Import ──────────────────────────────────────────

    @_require_unlocked
    def export_vault(self, path: Path, password: str) -> None:
        data = {}
        with self._vault_lock:
            for table in ("credentials", "user_profiles", "consent_log", "financial_data"):
                col_info = self._vault_conn.execute(f"PRAGMA table_info({table})").fetchall()
                col_names = [c[1] for c in col_info]
                rows = self._vault_conn.execute(f"SELECT * FROM {table}").fetchall()
                data[table] = [dict(zip(col_names, row)) for row in rows]
        payload = json.dumps(data).encode("utf-8")
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000)
        export_key = kdf.derive(password.encode("utf-8"))
        nonce = os.urandom(12)
        aesgcm = AESGCM(export_key)
        ct = aesgcm.encrypt(nonce, payload, None)
        Path(path).write_bytes(salt + nonce + ct)

    @_require_unlocked
    def import_vault(self, path: Path, password: str) -> None:
        raw = Path(path).read_bytes()
        salt, nonce, ct = raw[:16], raw[16:28], raw[28:]
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000)
        export_key = kdf.derive(password.encode("utf-8"))
        aesgcm = AESGCM(export_key)
        try:
            payload = aesgcm.decrypt(nonce, ct, None)
        except Exception as e:
            raise VaultAuthError(f"Wrong password or corrupted export: {e}") from e
        data = json.loads(payload.decode("utf-8"))
        with self._vault_lock:
            for table, rows in data.items():
                if not rows:
                    continue
                cols = list(rows[0].keys())
                placeholders = ", ".join("?" for _ in cols)
                col_str = ", ".join(cols)
                self._vault_conn.execute(f"DELETE FROM {table}")
                for row in rows:
                    self._vault_conn.execute(
                        f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})",
                        [row[c] for c in cols],
                    )
            self._vault_conn.commit()

    # ── Credentials ─────────────────────────────────────────────

    @_require_unlocked
    def store_credential(self, provider: str, account_id: str, token_type: str,
                        access_token: str, refresh_token: Optional[str] = None,
                        expires_at: Optional[float] = None,
                        scopes: Optional[list[str]] = None) -> str:
        cred_id = f"{provider}:{account_id}"
        key = self._keys["credentials"]
        now = time.time()
        enc_access = encrypt_field(access_token, key)
        enc_refresh = encrypt_field(refresh_token, key) if refresh_token else None
        scopes_json = json.dumps(scopes) if scopes else None
        with self._vault_lock:
            self._vault_conn.execute(
                """INSERT OR REPLACE INTO credentials
                   (id, provider, account_id, token_type, access_token,
                    refresh_token, expires_at, scopes, active, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (cred_id, provider, account_id, token_type, enc_access,
                 enc_refresh, expires_at, scopes_json, now, now),
            )
            self._vault_conn.commit()
        return cred_id

    @_require_unlocked
    def get_credential(self, provider: str,
                      account_id: Optional[str] = None) -> Optional[Credential]:
        with self._vault_lock:
            if account_id:
                row = self._vault_conn.execute(
                    "SELECT * FROM credentials WHERE id = ? AND active = 1",
                    (f"{provider}:{account_id}",),
                ).fetchone()
            else:
                row = self._vault_conn.execute(
                    "SELECT * FROM credentials WHERE provider = ? AND active = 1 LIMIT 1",
                    (provider,),
                ).fetchone()
        if not row:
            return None
        return self._row_to_credential(row)

    @_require_unlocked
    def list_credentials(self, provider: str) -> list[Credential]:
        with self._vault_lock:
            rows = self._vault_conn.execute(
                "SELECT * FROM credentials WHERE provider = ?", (provider,),
            ).fetchall()
        return [self._row_to_credential(r) for r in rows]

    @_require_unlocked
    def refresh_credential(self, credential_id: str, new_access_token: str,
                          new_expires_at: Optional[float] = None) -> None:
        lock = self._get_credential_lock(credential_id)
        with lock:
            key = self._keys["credentials"]
            enc_token = encrypt_field(new_access_token, key)
            now = time.time()
            with self._vault_lock:
                self._vault_conn.execute(
                    """UPDATE credentials SET access_token = ?, expires_at = ?,
                       updated_at = ? WHERE id = ?""",
                    (enc_token, new_expires_at or now + 3600, now, credential_id),
                )
                self._vault_conn.commit()

    @_require_unlocked
    def deactivate_credential(self, credential_id: str) -> None:
        with self._vault_lock:
            self._vault_conn.execute(
                "UPDATE credentials SET active = 0, updated_at = ? WHERE id = ?",
                (time.time(), credential_id),
            )
            self._vault_conn.commit()

    @_require_unlocked
    def delete_credential(self, credential_id: str) -> None:
        with self._vault_lock:
            self._vault_conn.execute(
                "DELETE FROM credentials WHERE id = ?", (credential_id,),
            )
            self._vault_conn.commit()
            self._vault_conn.execute("VACUUM")

    # ── Profiles ────────────────────────────────────────────────

    @_require_unlocked
    def store_profile(self, profile_id: str, display_name: Optional[str] = None,
                     email: Optional[str] = None, phone: Optional[str] = None,
                     metadata: Optional[dict] = None) -> None:
        key = self._keys["profiles"]
        now = time.time()
        enc_name = encrypt_field(display_name, key) if display_name else None
        enc_email = encrypt_field(email, key) if email else None
        enc_phone = encrypt_field(phone, key) if phone else None
        enc_meta = encrypt_field(json.dumps(metadata), key) if metadata else None
        with self._vault_lock:
            self._vault_conn.execute(
                """INSERT OR REPLACE INTO user_profiles
                   (id, display_name, email, phone, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (profile_id, enc_name, enc_email, enc_phone, enc_meta, now, now),
            )
            self._vault_conn.commit()

    @_require_unlocked
    def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        with self._vault_lock:
            row = self._vault_conn.execute(
                "SELECT * FROM user_profiles WHERE id = ?", (profile_id,),
            ).fetchone()
        if not row:
            return None
        key = self._keys["profiles"]
        return UserProfile(
            id=row[0],
            display_name=decrypt_field(row[1], key) if row[1] else None,
            email=decrypt_field(row[2], key) if row[2] else None,
            phone=decrypt_field(row[3], key) if row[3] else None,
            metadata=json.loads(decrypt_field(row[4], key)) if row[4] else None,
            created_at=row[5], updated_at=row[6],
        )

    # ── Consent ─────────────────────────────────────────────────

    @_require_unlocked
    def log_consent(self, provider: str, action: str,
                   scopes: Optional[list[str]] = None,
                   reason: Optional[str] = None) -> None:
        key = self._keys["consent"]
        enc_reason = encrypt_field(reason, key) if reason else None
        scopes_json = json.dumps(scopes) if scopes else None
        with self._vault_lock:
            self._vault_conn.execute(
                """INSERT INTO consent_log (provider, action, scopes, reason, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (provider, action, scopes_json, enc_reason, time.time()),
            )
            self._vault_conn.commit()

    @_require_unlocked
    def get_consent_history(self, provider: str) -> list[ConsentEntry]:
        with self._vault_lock:
            rows = self._vault_conn.execute(
                "SELECT * FROM consent_log WHERE provider = ? ORDER BY timestamp",
                (provider,),
            ).fetchall()
        key = self._keys["consent"]
        return [
            ConsentEntry(
                id=r[0], provider=r[1], action=r[2],
                scopes=json.loads(r[3]) if r[3] else None,
                reason=decrypt_field(r[4], key) if r[4] else None,
                timestamp=r[5],
            ) for r in rows
        ]

    @_require_unlocked
    def get_last_consent(self, provider: str) -> Optional[ConsentEntry]:
        with self._vault_lock:
            row = self._vault_conn.execute(
                "SELECT * FROM consent_log WHERE provider = ? ORDER BY timestamp DESC LIMIT 1",
                (provider,),
            ).fetchone()
        if not row:
            return None
        key = self._keys["consent"]
        return ConsentEntry(
            id=row[0], provider=row[1], action=row[2],
            scopes=json.loads(row[3]) if row[3] else None,
            reason=decrypt_field(row[4], key) if row[4] else None,
            timestamp=row[5],
        )

    # ── Financial ───────────────────────────────────────────────

    @_require_unlocked
    def store_financial(self, source: str, category: str, description: str,
                       amount: Optional[str] = None, currency: Optional[str] = None,
                       due_date: Optional[float] = None) -> int:
        key = self._keys["financial"]
        now = time.time()
        enc_desc = encrypt_field(description, key)
        enc_amount = encrypt_field(amount, key) if amount else None
        with self._vault_lock:
            cursor = self._vault_conn.execute(
                """INSERT INTO financial_data
                   (source, category, description, amount, currency, due_date,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (source, category, enc_desc, enc_amount, currency, due_date, now, now),
            )
            self._vault_conn.commit()
            return cursor.lastrowid

    @_require_unlocked
    def query_financial(self, status: Optional[str] = None,
                       due_before: Optional[float] = None,
                       category: Optional[str] = None) -> list[FinancialRecord]:
        query = "SELECT * FROM financial_data WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if due_before:
            query += " AND due_date IS NOT NULL AND due_date < ?"
            params.append(due_before)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY due_date"
        with self._vault_lock:
            rows = self._vault_conn.execute(query, params).fetchall()
        key = self._keys["financial"]
        return [
            FinancialRecord(
                id=r[0], source=r[1], category=r[2],
                description=decrypt_field(r[3], key),
                amount=decrypt_field(r[4], key) if r[4] else None,
                currency=r[5], due_date=r[6], status=r[7],
                reminded_at=r[8],
                raw_extract=decrypt_field(r[9], key) if r[9] else None,
                created_at=r[10], updated_at=r[11],
            ) for r in rows
        ]

    @_require_unlocked
    def update_financial(self, record_id: int, **kwargs) -> None:
        allowed = {"status", "reminded_at", "due_date"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [record_id]
        with self._vault_lock:
            self._vault_conn.execute(
                f"UPDATE financial_data SET {set_clause} WHERE id = ?", values,
            )
            self._vault_conn.commit()

    # ── Connection Status ───────────────────────────────────────

    @_require_unlocked
    def set_connection_status(self, provider: str, connected: bool,
                             label: Optional[str] = None,
                             mode: str = "always_on",
                             sync_interval: int = 300,
                             last_sync: Optional[float] = None) -> None:
        with self._cache_lock:
            self._cache_conn.execute(
                """INSERT OR REPLACE INTO connection_status
                   (provider, connected, display_label, connection_mode,
                    sync_interval, last_sync, last_sync_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (provider, int(connected), label, mode, sync_interval, last_sync, None),
            )
            self._cache_conn.commit()

    @_require_unlocked
    def get_connection_status(self, provider: str) -> Optional[ConnectionStatus]:
        with self._cache_lock:
            row = self._cache_conn.execute(
                "SELECT * FROM connection_status WHERE provider = ?", (provider,),
            ).fetchone()
        if not row:
            return None
        return ConnectionStatus(
            provider=row[0], connected=bool(row[1]),
            display_label=row[2], connection_mode=row[3],
            sync_interval=row[4], last_sync=row[5], last_sync_error=row[6],
        )

    @_require_unlocked
    def get_all_connections(self) -> list[ConnectionStatus]:
        with self._cache_lock:
            rows = self._cache_conn.execute("SELECT * FROM connection_status").fetchall()
        return [
            ConnectionStatus(
                provider=r[0], connected=bool(r[1]),
                display_label=r[2], connection_mode=r[3],
                sync_interval=r[4], last_sync=r[5], last_sync_error=r[6],
            ) for r in rows
        ]

    # ── Internal helpers ────────────────────────────────────────

    def _row_to_credential(self, row) -> Credential:
        key = self._keys["credentials"]
        return Credential(
            id=row[0], provider=row[1], account_id=row[2], token_type=row[3],
            access_token=decrypt_field(row[4], key),
            refresh_token=decrypt_field(row[5], key) if row[5] else None,
            expires_at=row[6],
            scopes=json.loads(row[7]) if row[7] else None,
            active=bool(row[8]), created_at=row[9], updated_at=row[10],
        )

    def _get_credential_lock(self, credential_id: str) -> threading.Lock:
        with self._credential_locks_lock:
            if credential_id not in self._credential_locks:
                self._credential_locks[credential_id] = threading.Lock()
            return self._credential_locks[credential_id]

    def _reencrypt_all(self, old_keys: dict[str, bytearray],
                      new_keys: dict[str, bytearray]) -> None:
        with self._vault_lock:
            # Re-encrypt credentials
            for r in self._vault_conn.execute("SELECT * FROM credentials").fetchall():
                old_k, new_k = old_keys["credentials"], new_keys["credentials"]
                new_access = encrypt_field(decrypt_field(r[4], old_k), new_k)
                new_refresh = encrypt_field(decrypt_field(r[5], old_k), new_k) if r[5] else None
                self._vault_conn.execute(
                    "UPDATE credentials SET access_token=?, refresh_token=? WHERE id=?",
                    (new_access, new_refresh, r[0]),
                )
            # Re-encrypt profiles
            for r in self._vault_conn.execute("SELECT * FROM user_profiles").fetchall():
                old_k, new_k = old_keys["profiles"], new_keys["profiles"]
                new_name = encrypt_field(decrypt_field(r[1], old_k), new_k) if r[1] else None
                new_email = encrypt_field(decrypt_field(r[2], old_k), new_k) if r[2] else None
                new_phone = encrypt_field(decrypt_field(r[3], old_k), new_k) if r[3] else None
                new_meta = encrypt_field(decrypt_field(r[4], old_k), new_k) if r[4] else None
                self._vault_conn.execute(
                    "UPDATE user_profiles SET display_name=?, email=?, phone=?, metadata=? WHERE id=?",
                    (new_name, new_email, new_phone, new_meta, r[0]),
                )
            # Re-encrypt financial
            for r in self._vault_conn.execute("SELECT * FROM financial_data").fetchall():
                old_k, new_k = old_keys["financial"], new_keys["financial"]
                new_desc = encrypt_field(decrypt_field(r[3], old_k), new_k)
                new_amt = encrypt_field(decrypt_field(r[4], old_k), new_k) if r[4] else None
                new_raw = encrypt_field(decrypt_field(r[9], old_k), new_k) if r[9] else None
                self._vault_conn.execute(
                    "UPDATE financial_data SET description=?, amount=?, raw_extract=? WHERE id=?",
                    (new_desc, new_amt, new_raw, r[0]),
                )
            # Re-encrypt consent reasons
            for r in self._vault_conn.execute("SELECT * FROM consent_log").fetchall():
                if r[4]:
                    old_k, new_k = old_keys["consent"], new_keys["consent"]
                    new_reason = encrypt_field(decrypt_field(r[4], old_k), new_k)
                    self._vault_conn.execute(
                        "UPDATE consent_log SET reason=? WHERE id=?", (new_reason, r[0]),
                    )
            self._vault_conn.commit()
