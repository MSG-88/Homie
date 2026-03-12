"""OS keyring integration with file-based fallback and optional password layer.

Stores the master key in the OS keyring (Windows Credential Manager, macOS
Keychain, Linux Secret Service). Falls back to an encrypted file if the keyring
is unavailable. Supports an optional password layer that encrypts the master key
with PBKDF2 before storage.

Rate limiting: persists failed attempt count in vault.meta.json to survive restarts.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from homie_core.vault.exceptions import VaultAuthError, RateLimitError

_SERVICE_NAME = "homie-vault"
_KEY_USERNAME = "master-key"
_ENCRYPTED_KEY_USERNAME = "encrypted-master-key"
_PBKDF2_ITERATIONS = 600_000
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 60.0


def _keyring_available() -> bool:
    """Check if the OS keyring is usable."""
    try:
        import keyring
        keyring.get_password(_SERVICE_NAME, "__probe__")
        return True
    except Exception:
        return False


class KeyringBackend:
    """Manages master key storage in OS keyring or file fallback."""

    def __init__(self, fallback_dir: Optional[Path] = None):
        self._fallback_dir = fallback_dir or Path.home() / ".homie" / "vault"
        self._fallback_dir.mkdir(parents=True, exist_ok=True)
        self._use_keyring = _keyring_available()
        self._meta_path = self._fallback_dir / "vault.meta.json"

    def store_master_key(self, key: bytearray) -> None:
        """Store the raw master key."""
        encoded = base64.b64encode(bytes(key)).decode("ascii")
        if self._use_keyring:
            import keyring
            keyring.set_password(_SERVICE_NAME, _KEY_USERNAME, encoded)
        else:
            keyfile = self._fallback_dir / ".keyfile"
            keyfile.write_text(encoded, encoding="ascii")
            try:
                keyfile.chmod(0o600)
            except OSError:
                pass

    def retrieve_master_key(self, password: Optional[str] = None) -> Optional[bytearray]:
        """Retrieve master key. If password-protected, decrypts with password.

        Raises VaultAuthError on wrong password.
        Raises RateLimitError after too many failed attempts.
        """
        if password is not None:
            self._check_rate_limit()

        if self.has_password() and password is not None:
            return self._retrieve_with_password(password)

        encoded = self._read_raw_key()
        if encoded is None:
            return None
        try:
            return bytearray(base64.b64decode(encoded))
        except Exception:
            return None

    def delete_master_key(self) -> None:
        """Remove the master key from storage."""
        if self._use_keyring:
            import keyring
            for username in (_KEY_USERNAME, _ENCRYPTED_KEY_USERNAME):
                try:
                    keyring.delete_password(_SERVICE_NAME, username)
                except Exception:
                    pass
        else:
            for fname in (".keyfile", ".keyfile.pw"):
                f = self._fallback_dir / fname
                if f.exists():
                    f.unlink()

    def set_password(self, password: str, master_key: bytearray) -> None:
        """Enable password protection. Encrypts master key with PBKDF2-derived key."""
        salt = os.urandom(16)
        password_key = self._derive_password_key(password, salt)
        nonce = os.urandom(12)
        aesgcm = AESGCM(password_key)
        encrypted = aesgcm.encrypt(nonce, bytes(master_key), None)
        blob = base64.b64encode(salt + nonce + encrypted).decode("ascii")

        if self._use_keyring:
            import keyring
            keyring.set_password(_SERVICE_NAME, _ENCRYPTED_KEY_USERNAME, blob)
        else:
            pw_file = self._fallback_dir / ".keyfile.pw"
            pw_file.write_text(blob, encoding="ascii")
            try:
                pw_file.chmod(0o600)
            except OSError:
                pass

    def has_password(self) -> bool:
        """Check if a password layer is configured."""
        if self._use_keyring:
            try:
                import keyring
                return keyring.get_password(_SERVICE_NAME, _ENCRYPTED_KEY_USERNAME) is not None
            except Exception:
                return False
        else:
            return (self._fallback_dir / ".keyfile.pw").exists()

    def _read_raw_key(self) -> Optional[str]:
        if self._use_keyring:
            try:
                import keyring
                return keyring.get_password(_SERVICE_NAME, _KEY_USERNAME)
            except Exception:
                return None
        else:
            keyfile = self._fallback_dir / ".keyfile"
            if keyfile.exists():
                return keyfile.read_text(encoding="ascii").strip()
            return None

    def _retrieve_with_password(self, password: str) -> bytearray:
        """Decrypt master key using password."""
        if self._use_keyring:
            import keyring
            blob_str = keyring.get_password(_SERVICE_NAME, _ENCRYPTED_KEY_USERNAME)
        else:
            pw_file = self._fallback_dir / ".keyfile.pw"
            blob_str = pw_file.read_text(encoding="ascii").strip() if pw_file.exists() else None

        if not blob_str:
            raise VaultAuthError("No password-encrypted key found")

        try:
            blob = base64.b64decode(blob_str)
            salt = blob[:16]
            nonce = blob[16:28]
            encrypted = blob[28:]
            password_key = self._derive_password_key(password, salt)
            aesgcm = AESGCM(password_key)
            master_key = aesgcm.decrypt(nonce, encrypted, None)
            self._reset_rate_limit()
            return bytearray(master_key)
        except VaultAuthError:
            raise
        except Exception as e:
            self._record_failed_attempt()
            raise VaultAuthError(f"Wrong password or corrupted key: {e}") from e

    @staticmethod
    def _derive_password_key(password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=_PBKDF2_ITERATIONS,
        )
        return kdf.derive(password.encode("utf-8"))

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            try:
                return json.loads(self._meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_meta(self, meta: dict) -> None:
        self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            self._meta_path.chmod(0o600)
        except OSError:
            pass

    def _check_rate_limit(self) -> None:
        meta = self._load_meta()
        failures = meta.get("failed_attempts", 0)
        lockout_until = meta.get("lockout_until", 0)
        if failures >= _MAX_ATTEMPTS and time.time() < lockout_until:
            retry_after = lockout_until - time.time()
            raise RateLimitError(
                f"Too many failed attempts. Try again in {retry_after:.0f}s.",
                retry_after=retry_after,
            )
        if failures >= _MAX_ATTEMPTS and time.time() >= lockout_until:
            meta["failed_attempts"] = 0
            meta.pop("lockout_until", None)
            self._save_meta(meta)

    def _record_failed_attempt(self) -> None:
        meta = self._load_meta()
        meta["failed_attempts"] = meta.get("failed_attempts", 0) + 1
        if meta["failed_attempts"] >= _MAX_ATTEMPTS:
            meta["lockout_until"] = time.time() + _LOCKOUT_SECONDS
        self._save_meta(meta)

    def _reset_rate_limit(self) -> None:
        meta = self._load_meta()
        meta["failed_attempts"] = 0
        meta.pop("lockout_until", None)
        self._save_meta(meta)
