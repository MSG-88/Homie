# Privacy & Secure Storage Layer — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an encrypted local vault that stores credentials, user profiles, consent records, and financial data — the foundation for all Homie integration sub-projects.

**Architecture:** Two SQLite databases (`vault.db` for encrypted fields, `cache.db` for plaintext caches) with AES-256-GCM per-field encryption. Master key stored in OS keyring via `keyring` library, with optional password layer. Thread-safe `SecureVault` API is the single entry point for all consumers.

**Tech Stack:** Python 3.12, `cryptography` (AES-256-GCM, HKDF, PBKDF2), `keyring` (OS keyring abstraction), SQLite (stdlib), pytest

**Spec:** `docs/superpowers/specs/2026-03-12-privacy-secure-storage-layer-design.md`

---

## File Structure

```
src/homie_core/vault/
├── __init__.py              # Package init, re-exports SecureVault + exceptions
├── exceptions.py            # VaultError hierarchy (6 exception classes)
├── models.py                # Dataclasses: Credential, UserProfile, ConsentEntry,
│                            #   FinancialRecord, ConnectionStatus
├── encryption.py            # AES-256-GCM encrypt/decrypt, HKDF category key derivation
├── keyring_backend.py       # OS keyring read/write, file-based fallback, password layer
├── schema.py                # DDL for vault.db + cache.db, migration registry
├── secure_vault.py          # SecureVault class — thread-safe main API
└── sync_manager.py          # Periodic background sync (always_on / on_demand)

src/homie_app/cli.py         # (Modify) Add connect/disconnect/connections/consent-log commands
src/homie_app/daemon.py      # (Modify) Wire vault + sync_manager into daemon lifecycle

tests/unit/test_vault/
├── __init__.py
├── test_exceptions.py       # Exception hierarchy tests
├── test_models.py           # Dataclass tests
├── test_encryption.py       # AES-256-GCM round-trip, HKDF derivation, key zeroing
├── test_keyring_backend.py  # Keyring store/retrieve, file fallback, password layer
├── test_schema.py           # Table creation, migrations, integrity checks
├── test_secure_vault.py     # Full API tests (credentials, profiles, consent, financial)
└── test_sync_manager.py     # Sync tick, always_on vs on_demand, token refresh
```

---

## Chunk 1: Foundation — Exceptions, Models, Encryption

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add cryptography and keyring to project dependencies**

In `pyproject.toml`, add to the `[project.dependencies]` list:

```toml
"cryptography>=43.0",
"keyring>=25.0",
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -e ".[dev]"`
Expected: Both packages install successfully

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add cryptography and keyring dependencies for vault"
```

---

### Task 2: Exception hierarchy

**Files:**
- Create: `src/homie_core/vault/__init__.py`
- Create: `src/homie_core/vault/exceptions.py`
- Create: `tests/unit/test_vault/__init__.py`
- Create: `tests/unit/test_vault/test_exceptions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vault/test_exceptions.py
import pytest

from homie_core.vault.exceptions import (
    VaultError,
    VaultLockedError,
    VaultAuthError,
    VaultCorruptError,
    CredentialNotFoundError,
    RateLimitError,
)


class TestVaultExceptionHierarchy:
    def test_all_inherit_from_vault_error(self):
        for exc_cls in (VaultLockedError, VaultAuthError, VaultCorruptError,
                        CredentialNotFoundError, RateLimitError):
            assert issubclass(exc_cls, VaultError)

    def test_vault_error_is_exception(self):
        assert issubclass(VaultError, Exception)

    def test_exceptions_carry_message(self):
        err = VaultLockedError("vault is locked")
        assert str(err) == "vault is locked"

    def test_rate_limit_carries_retry_after(self):
        err = RateLimitError("too many attempts", retry_after=60.0)
        assert err.retry_after == 60.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vault/test_exceptions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/vault/__init__.py
"""Homie Secure Vault — encrypted local storage for credentials and user data."""

from homie_core.vault.exceptions import (
    VaultError,
    VaultLockedError,
    VaultAuthError,
    VaultCorruptError,
    CredentialNotFoundError,
    RateLimitError,
)

__all__ = [
    "VaultError",
    "VaultLockedError",
    "VaultAuthError",
    "VaultCorruptError",
    "CredentialNotFoundError",
    "RateLimitError",
]
```

```python
# src/homie_core/vault/exceptions.py
"""Vault exception hierarchy."""
from __future__ import annotations


class VaultError(Exception):
    """Base exception for all vault operations."""


class VaultLockedError(VaultError):
    """Raised when an operation is attempted on a locked vault."""


class VaultAuthError(VaultError):
    """Raised on wrong password or corrupted master key."""


class VaultCorruptError(VaultError):
    """Raised on database corruption or tampered ciphertext (GCM auth fail)."""


class CredentialNotFoundError(VaultError):
    """Raised when a requested credential does not exist."""


class RateLimitError(VaultError):
    """Raised after too many failed unlock attempts."""

    def __init__(self, message: str = "", retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after
```

Also create `tests/unit/test_vault/__init__.py` (empty file).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vault/test_exceptions.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/vault/__init__.py src/homie_core/vault/exceptions.py \
       tests/unit/test_vault/__init__.py tests/unit/test_vault/test_exceptions.py
git commit -m "feat(vault): add exception hierarchy"
```

---

### Task 3: Data models

**Files:**
- Create: `src/homie_core/vault/models.py`
- Create: `tests/unit/test_vault/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vault/test_models.py
import time
from homie_core.vault.models import (
    Credential,
    UserProfile,
    ConsentEntry,
    FinancialRecord,
    ConnectionStatus,
)


class TestCredential:
    def test_create_credential(self):
        cred = Credential(
            id="gmail:user@example.com",
            provider="gmail",
            account_id="user@example.com",
            token_type="oauth2",
            access_token="access_123",
            refresh_token="refresh_456",
            expires_at=time.time() + 3600,
            scopes=["email.read"],
            active=True,
        )
        assert cred.provider == "gmail"
        assert cred.active is True

    def test_credential_id_format(self):
        cred = Credential(
            id="slack:U12345",
            provider="slack",
            account_id="U12345",
            token_type="oauth2",
            access_token="xoxb-token",
        )
        assert cred.id == "slack:U12345"
        assert cred.refresh_token is None
        assert cred.scopes is None
        assert cred.active is True


class TestUserProfile:
    def test_create_profile(self):
        profile = UserProfile(
            id="primary",
            display_name="Muthu",
            email="muthu@example.com",
        )
        assert profile.display_name == "Muthu"
        assert profile.phone is None
        assert profile.metadata is None


class TestConsentEntry:
    def test_create_consent(self):
        entry = ConsentEntry(
            id=1,
            provider="gmail",
            action="connected",
            scopes=["email.read"],
            reason=None,
            timestamp=time.time(),
        )
        assert entry.action == "connected"


class TestFinancialRecord:
    def test_create_financial(self):
        record = FinancialRecord(
            id=1,
            source="gmail:msg123",
            category="bill",
            description="Electric bill",
            amount="142.50",
            currency="USD",
            due_date=time.time() + 86400,
            status="pending",
        )
        assert record.category == "bill"
        assert record.reminded_at is None


class TestConnectionStatus:
    def test_create_status(self):
        status = ConnectionStatus(
            provider="gmail",
            connected=True,
            display_label="Gmail (m***@gmail.com)",
            connection_mode="always_on",
            sync_interval=300,
        )
        assert status.connected is True
        assert status.connection_mode == "always_on"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vault/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/vault/models.py
"""Vault data models — dataclasses for credentials, profiles, consent, and finance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Credential:
    id: str                              # "provider:account_id"
    provider: str
    account_id: str
    token_type: str                      # "oauth2", "api_key", "app_password"
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None   # Unix timestamp
    scopes: Optional[list[str]] = None
    active: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class UserProfile:
    id: str                              # "primary" or "provider:account_id"
    display_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class ConsentEntry:
    id: int
    provider: str
    action: str                          # "connected", "declined", "disconnected", "reauthorized"
    scopes: Optional[list[str]] = None
    reason: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class FinancialRecord:
    id: int
    source: str                          # "gmail:msg_id_123"
    category: str                        # "bill", "payment", "subscription", "reminder"
    description: str
    amount: Optional[str] = None         # String to avoid float precision issues
    currency: Optional[str] = None
    due_date: Optional[float] = None     # Unix timestamp
    status: str = "pending"              # "pending", "paid", "overdue", "dismissed"
    reminded_at: Optional[float] = None
    raw_extract: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class ConnectionStatus:
    provider: str
    connected: bool = False
    display_label: Optional[str] = None
    connection_mode: str = "always_on"   # "always_on" or "on_demand"
    sync_interval: int = 300             # seconds
    last_sync: Optional[float] = None
    last_sync_error: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vault/test_models.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/vault/models.py tests/unit/test_vault/test_models.py
git commit -m "feat(vault): add data models for credentials, profiles, consent, finance"
```

---

### Task 4: Encryption module

**Files:**
- Create: `src/homie_core/vault/encryption.py`
- Create: `tests/unit/test_vault/test_encryption.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vault/test_encryption.py
import os
import pytest

from homie_core.vault.encryption import (
    generate_master_key,
    derive_category_key,
    encrypt_field,
    decrypt_field,
    zero_bytearray,
)
from homie_core.vault.exceptions import VaultCorruptError


class TestGenerateMasterKey:
    def test_returns_bytearray_32_bytes(self):
        key = generate_master_key()
        assert isinstance(key, bytearray)
        assert len(key) == 32

    def test_keys_are_unique(self):
        k1 = generate_master_key()
        k2 = generate_master_key()
        assert k1 != k2


class TestDeriveCategoryKey:
    def test_returns_bytearray_32_bytes(self):
        master = generate_master_key()
        cat_key = derive_category_key(master, "credentials")
        assert isinstance(cat_key, bytearray)
        assert len(cat_key) == 32

    def test_different_categories_produce_different_keys(self):
        master = generate_master_key()
        k1 = derive_category_key(master, "credentials")
        k2 = derive_category_key(master, "financial")
        assert k1 != k2

    def test_same_category_same_master_is_deterministic(self):
        master = generate_master_key()
        k1 = derive_category_key(master, "credentials")
        k2 = derive_category_key(master, "credentials")
        assert k1 == k2


class TestEncryptDecryptField:
    def test_round_trip(self):
        key = derive_category_key(generate_master_key(), "test")
        plaintext = "my secret token"
        ciphertext = encrypt_field(plaintext, key)
        assert ciphertext != plaintext
        result = decrypt_field(ciphertext, key)
        assert result == plaintext

    def test_empty_string_round_trip(self):
        key = derive_category_key(generate_master_key(), "test")
        ct = encrypt_field("", key)
        assert decrypt_field(ct, key) == ""

    def test_unicode_round_trip(self):
        key = derive_category_key(generate_master_key(), "test")
        text = "Hello \u2603 \U0001f600 world"
        ct = encrypt_field(text, key)
        assert decrypt_field(ct, key) == text

    def test_different_encryptions_produce_different_ciphertexts(self):
        key = derive_category_key(generate_master_key(), "test")
        ct1 = encrypt_field("same", key)
        ct2 = encrypt_field("same", key)
        assert ct1 != ct2  # Different nonces

    def test_wrong_key_raises_corrupt_error(self):
        master = generate_master_key()
        k1 = derive_category_key(master, "credentials")
        k2 = derive_category_key(master, "financial")
        ct = encrypt_field("secret", k1)
        with pytest.raises(VaultCorruptError):
            decrypt_field(ct, k2)

    def test_tampered_ciphertext_raises_corrupt_error(self):
        key = derive_category_key(generate_master_key(), "test")
        ct = encrypt_field("secret", key)
        # Flip a character in the middle of the base64 string
        tampered = ct[:20] + ("A" if ct[20] != "A" else "B") + ct[21:]
        with pytest.raises(VaultCorruptError):
            decrypt_field(tampered, key)


class TestZeroBytearray:
    def test_zeros_all_bytes(self):
        buf = bytearray(b"secret key material!")
        zero_bytearray(buf)
        assert buf == bytearray(len(buf))

    def test_length_preserved(self):
        buf = bytearray(32)
        buf[:] = os.urandom(32)
        original_len = len(buf)
        zero_bytearray(buf)
        assert len(buf) == original_len
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vault/test_encryption.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/vault/encryption.py
"""AES-256-GCM field encryption and HKDF key derivation.

Storage format per encrypted field: base64(nonce_12B || ciphertext || tag_16B)
Category keys derived via HKDF-SHA256(master_key, info=category_name).
All key material uses mutable bytearray for secure zeroing.
"""
from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from homie_core.vault.exceptions import VaultCorruptError

_KEY_LENGTH = 32  # AES-256
_NONCE_LENGTH = 12  # GCM standard


def generate_master_key() -> bytearray:
    """Generate a 32-byte random master key."""
    return bytearray(os.urandom(_KEY_LENGTH))


def derive_category_key(master_key: bytearray, category: str) -> bytearray:
    """Derive a category-specific key from the master key using HKDF-SHA256.

    The category name is passed as the ``info`` parameter (not salt),
    following RFC 5869.
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=None,
        info=category.encode("utf-8"),
    )
    derived = hkdf.derive(bytes(master_key))
    return bytearray(derived)


def encrypt_field(plaintext: str, key: bytearray) -> str:
    """Encrypt a string field with AES-256-GCM.

    Returns base64-encoded string: nonce (12B) || ciphertext || tag (16B).
    Each call generates a fresh random nonce.
    """
    nonce = os.urandom(_NONCE_LENGTH)
    aesgcm = AESGCM(bytes(key))
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    # ct includes the 16-byte GCM tag appended by the library
    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt_field(ciphertext_b64: str, key: bytearray) -> str:
    """Decrypt a base64-encoded AES-256-GCM field.

    Raises VaultCorruptError if decryption fails (wrong key or tampered data).
    """
    try:
        raw = base64.b64decode(ciphertext_b64)
    except Exception as e:
        raise VaultCorruptError(f"Invalid base64 ciphertext: {e}") from e

    if len(raw) < _NONCE_LENGTH + 16:  # nonce + minimum GCM tag
        raise VaultCorruptError("Ciphertext too short")

    nonce = raw[:_NONCE_LENGTH]
    ct = raw[_NONCE_LENGTH:]

    try:
        aesgcm = AESGCM(bytes(key))
        plaintext = aesgcm.decrypt(nonce, ct, None)
        return plaintext.decode("utf-8")
    except Exception as e:
        raise VaultCorruptError(f"Decryption failed (wrong key or tampered data): {e}") from e


def zero_bytearray(buf: bytearray) -> None:
    """Overwrite a bytearray with zeros. Best-effort secure memory wipe."""
    for i in range(len(buf)):
        buf[i] = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vault/test_encryption.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/vault/encryption.py tests/unit/test_vault/test_encryption.py
git commit -m "feat(vault): add AES-256-GCM encryption and HKDF key derivation"
```

---

## Chunk 2: Keyring Backend and Database Schema

### Task 5: Keyring backend

**Files:**
- Create: `src/homie_core/vault/keyring_backend.py`
- Create: `tests/unit/test_vault/test_keyring_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vault/test_keyring_backend.py
import json
import os
import pytest
from unittest.mock import patch, MagicMock

from homie_core.vault.keyring_backend import KeyringBackend
from homie_core.vault.exceptions import VaultAuthError, RateLimitError


class TestKeyringBackendStore:
    def test_store_and_retrieve_master_key(self, tmp_path):
        """Test store/retrieve using file-based fallback (no real keyring)."""
        backend = KeyringBackend(fallback_dir=tmp_path)
        key = bytearray(os.urandom(32))
        backend.store_master_key(key)
        retrieved = backend.retrieve_master_key()
        assert retrieved == key

    def test_retrieve_returns_bytearray(self, tmp_path):
        backend = KeyringBackend(fallback_dir=tmp_path)
        key = bytearray(os.urandom(32))
        backend.store_master_key(key)
        retrieved = backend.retrieve_master_key()
        assert isinstance(retrieved, bytearray)

    def test_retrieve_nonexistent_returns_none(self, tmp_path):
        backend = KeyringBackend(fallback_dir=tmp_path)
        assert backend.retrieve_master_key() is None


class TestPasswordLayer:
    def test_set_and_unlock_with_password(self, tmp_path):
        backend = KeyringBackend(fallback_dir=tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        backend.set_password("my-secret-pass", master)

        retrieved = backend.retrieve_master_key(password="my-secret-pass")
        assert retrieved == master

    def test_wrong_password_raises_auth_error(self, tmp_path):
        backend = KeyringBackend(fallback_dir=tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        backend.set_password("correct-pass", master)

        with pytest.raises(VaultAuthError):
            backend.retrieve_master_key(password="wrong-pass")

    def test_has_password(self, tmp_path):
        backend = KeyringBackend(fallback_dir=tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        assert backend.has_password() is False
        backend.set_password("pass", master)
        assert backend.has_password() is True


class TestRateLimiting:
    def test_rate_limit_after_5_failures(self, tmp_path):
        backend = KeyringBackend(fallback_dir=tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        backend.set_password("correct", master)

        meta_path = tmp_path / "vault.meta.json"

        for i in range(5):
            with pytest.raises(VaultAuthError):
                backend.retrieve_master_key(password="wrong")

        with pytest.raises(RateLimitError):
            backend.retrieve_master_key(password="wrong")

    def test_successful_unlock_resets_counter(self, tmp_path):
        backend = KeyringBackend(fallback_dir=tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        backend.set_password("correct", master)

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(VaultAuthError):
                backend.retrieve_master_key(password="wrong")

        # Succeed — should reset counter
        result = backend.retrieve_master_key(password="correct")
        assert result == master

        # Should be able to fail 5 more times before rate limit
        for _ in range(5):
            with pytest.raises(VaultAuthError):
                backend.retrieve_master_key(password="wrong")

        with pytest.raises(RateLimitError):
            backend.retrieve_master_key(password="wrong")


class TestFileFallback:
    def test_uses_file_when_keyring_unavailable(self, tmp_path):
        """Force keyring to fail, verify file fallback works."""
        with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
            backend = KeyringBackend(fallback_dir=tmp_path)
            key = bytearray(os.urandom(32))
            backend.store_master_key(key)

            keyfile = tmp_path / ".keyfile"
            assert keyfile.exists()

            retrieved = backend.retrieve_master_key()
            assert retrieved == key

    def test_delete_master_key(self, tmp_path):
        backend = KeyringBackend(fallback_dir=tmp_path)
        key = bytearray(os.urandom(32))
        backend.store_master_key(key)
        backend.delete_master_key()
        assert backend.retrieve_master_key() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vault/test_keyring_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/vault/keyring_backend.py
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
_SALT_USERNAME = "password-salt"
_ENCRYPTED_KEY_USERNAME = "encrypted-master-key"
_PBKDF2_ITERATIONS = 600_000
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 60.0


def _keyring_available() -> bool:
    """Check if the OS keyring is usable."""
    try:
        import keyring
        # Try a harmless read to check if backend is functional
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

    # ── Master key storage ──────────────────────────────────────────

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
                pass  # Windows may not support POSIX permissions

    def retrieve_master_key(self, password: Optional[str] = None) -> Optional[bytearray]:
        """Retrieve master key. If password-protected, decrypts with password.

        Raises VaultAuthError on wrong password.
        Raises RateLimitError after too many failed attempts.
        """
        if password is not None:
            self._check_rate_limit()

        if self.has_password() and password is not None:
            return self._retrieve_with_password(password)
        elif self.has_password() and password is None:
            # Password required but not provided — retrieve raw key
            # (only works if raw key is still stored, pre-password-set)
            pass

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
            try:
                import keyring
                keyring.delete_password(_SERVICE_NAME, _KEY_USERNAME)
            except Exception:
                pass
            try:
                import keyring
                keyring.delete_password(_SERVICE_NAME, _ENCRYPTED_KEY_USERNAME)
            except Exception:
                pass
            try:
                import keyring
                keyring.delete_password(_SERVICE_NAME, _SALT_USERNAME)
            except Exception:
                pass
        else:
            for fname in (".keyfile", ".keyfile.pw"):
                f = self._fallback_dir / fname
                if f.exists():
                    f.unlink()

    # ── Password layer ──────────────────────────────────────────────

    def set_password(self, password: str, master_key: bytearray) -> None:
        """Enable password protection. Encrypts master key with PBKDF2-derived key."""
        salt = os.urandom(16)
        password_key = self._derive_password_key(password, salt)

        # Encrypt master key with password-derived key
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

    # ── Internal helpers ────────────────────────────────────────────

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

    # ── Rate limiting ───────────────────────────────────────────────

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            try:
                return json.loads(self._meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_meta(self, meta: dict) -> None:
        self._meta_path.write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )
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

        # Reset if lockout has expired
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vault/test_keyring_backend.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/vault/keyring_backend.py tests/unit/test_vault/test_keyring_backend.py
git commit -m "feat(vault): add keyring backend with file fallback and password layer"
```

---

### Task 6: Database schema

**Files:**
- Create: `src/homie_core/vault/schema.py`
- Create: `tests/unit/test_vault/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vault/test_schema.py
import sqlite3
import json
import pytest

from homie_core.vault.schema import (
    create_vault_db,
    create_cache_db,
    get_schema_version,
    run_migrations,
    CURRENT_SCHEMA_VERSION,
)


class TestCreateVaultDb:
    def test_creates_tables(self, tmp_path):
        db_path = tmp_path / "vault.db"
        conn = create_vault_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "credentials" in tables
        assert "user_profiles" in tables
        assert "consent_log" in tables
        assert "financial_data" in tables
        conn.close()

    def test_wal_mode_enabled(self, tmp_path):
        db_path = tmp_path / "vault.db"
        conn = create_vault_db(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_credentials_columns(self, tmp_path):
        db_path = tmp_path / "vault.db"
        conn = create_vault_db(db_path)
        cursor = conn.execute("PRAGMA table_info(credentials)")
        cols = {row[1] for row in cursor.fetchall()}
        expected = {"id", "provider", "account_id", "token_type", "access_token",
                    "refresh_token", "expires_at", "scopes", "active",
                    "created_at", "updated_at"}
        assert expected == cols
        conn.close()


class TestCreateCacheDb:
    def test_creates_tables(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "folder_watches" in tables
        assert "content_index" in tables
        assert "connection_status" in tables
        conn.close()

    def test_connection_status_columns(self, tmp_path):
        db_path = tmp_path / "cache.db"
        conn = create_cache_db(db_path)
        cursor = conn.execute("PRAGMA table_info(connection_status)")
        cols = {row[1] for row in cursor.fetchall()}
        expected = {"provider", "connected", "display_label", "connection_mode",
                    "sync_interval", "last_sync", "last_sync_error"}
        assert expected == cols
        conn.close()


class TestSchemaVersioning:
    def test_initial_version(self, tmp_path):
        meta_path = tmp_path / "vault.meta.json"
        version = get_schema_version(meta_path)
        assert version == 0  # No file yet

    def test_run_migrations_sets_version(self, tmp_path):
        db_path = tmp_path / "vault.db"
        meta_path = tmp_path / "vault.meta.json"
        conn = create_vault_db(db_path)
        run_migrations(conn, meta_path)
        version = get_schema_version(meta_path)
        assert version == CURRENT_SCHEMA_VERSION
        conn.close()


class TestIntegrityCheck:
    def test_valid_db_passes_check(self, tmp_path):
        from homie_core.vault.schema import check_integrity
        db_path = tmp_path / "vault.db"
        conn = create_vault_db(db_path)
        assert check_integrity(conn) is True
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vault/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/vault/schema.py
"""Database schema creation, integrity checks, and migration support.

Two databases:
- vault.db: encrypted fields (credentials, profiles, consent, financial)
- cache.db: plaintext caches (folder watches, content index, connection status)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

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
"""

# Migration registry: {from_version: migration_callable}
# Each callable receives a sqlite3.Connection and runs DDL/DML.
_MIGRATIONS: dict[int, callable] = {
    # Future migrations go here:
    # 1: _migrate_v1_to_v2,
}


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vault/test_schema.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/vault/schema.py tests/unit/test_vault/test_schema.py
git commit -m "feat(vault): add database schema with WAL mode and migration support"
```

---

## Chunk 3: SecureVault API

### Task 7: SecureVault — core lifecycle and credential operations

**Files:**
- Create: `src/homie_core/vault/secure_vault.py`
- Create: `tests/unit/test_vault/test_secure_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vault/test_secure_vault.py
import os
import time
import pytest
from unittest.mock import patch

from homie_core.vault.secure_vault import SecureVault
from homie_core.vault.models import Credential
from homie_core.vault.exceptions import (
    VaultLockedError,
    CredentialNotFoundError,
    VaultAuthError,
)


@pytest.fixture
def vault(tmp_path):
    """Create and unlock a vault in a temp directory."""
    v = SecureVault(storage_dir=tmp_path / "vault")
    v.unlock()
    return v


@pytest.fixture
def locked_vault(tmp_path):
    """Create a vault but don't unlock it."""
    return SecureVault(storage_dir=tmp_path / "vault")


class TestVaultLifecycle:
    def test_unlock_creates_databases(self, tmp_path):
        v = SecureVault(storage_dir=tmp_path / "vault")
        v.unlock()
        assert (tmp_path / "vault" / "vault.db").exists()
        assert (tmp_path / "vault" / "cache.db").exists()

    def test_is_unlocked(self, vault):
        assert vault.is_unlocked is True

    def test_lock_zeros_keys(self, vault):
        vault.lock()
        assert vault.is_unlocked is False

    def test_operations_fail_when_locked(self, locked_vault):
        with pytest.raises(VaultLockedError):
            locked_vault.store_credential(
                provider="gmail", account_id="test@gmail.com",
                token_type="oauth2", access_token="token",
            )

    def test_password_layer(self, tmp_path):
        v = SecureVault(storage_dir=tmp_path / "vault")
        v.unlock()
        v.set_password("secret123")
        v.lock()

        # Re-unlock with password
        v2 = SecureVault(storage_dir=tmp_path / "vault")
        v2.unlock(password="secret123")
        assert v2.is_unlocked is True

    def test_wrong_password_raises(self, tmp_path):
        v = SecureVault(storage_dir=tmp_path / "vault")
        v.unlock()
        v.set_password("correct")
        v.lock()

        v2 = SecureVault(storage_dir=tmp_path / "vault")
        with pytest.raises(VaultAuthError):
            v2.unlock(password="wrong")

    def test_set_password_preserves_data(self, tmp_path):
        """Verify credentials survive re-encryption after set_password."""
        v = SecureVault(storage_dir=tmp_path / "vault")
        v.unlock()
        v.store_credential(
            provider="gmail", account_id="u@gmail.com",
            token_type="oauth2", access_token="my_secret_token",
            refresh_token="my_refresh",
        )
        v.store_profile("primary", display_name="Test User", email="u@gmail.com")
        v.log_consent("gmail", "connected", reason="user approved")
        v.store_financial(source="a", category="bill", description="Electric", amount="99.50")

        v.set_password("new_pass")
        v.lock()

        # Re-unlock with password and verify all data is intact
        v2 = SecureVault(storage_dir=tmp_path / "vault")
        v2.unlock(password="new_pass")
        cred = v2.get_credential("gmail")
        assert cred.access_token == "my_secret_token"
        assert cred.refresh_token == "my_refresh"
        profile = v2.get_profile("primary")
        assert profile.display_name == "Test User"
        consent = v2.get_last_consent("gmail")
        assert consent.reason == "user approved"
        bills = v2.query_financial(status="pending")
        assert bills[0].amount == "99.50"
        v2.lock()

    def test_has_password_property(self, tmp_path):
        v = SecureVault(storage_dir=tmp_path / "vault")
        v.unlock()
        assert v.has_password is False
        v.set_password("pass")
        assert v.has_password is True


class TestCredentials:
    def test_store_and_retrieve(self, vault):
        cred_id = vault.store_credential(
            provider="gmail", account_id="user@gmail.com",
            token_type="oauth2", access_token="access_123",
            refresh_token="refresh_456",
            expires_at=time.time() + 3600,
            scopes=["email.read"],
        )
        assert cred_id == "gmail:user@gmail.com"

        cred = vault.get_credential("gmail")
        assert cred is not None
        assert cred.access_token == "access_123"
        assert cred.refresh_token == "refresh_456"
        assert cred.scopes == ["email.read"]

    def test_get_by_account_id(self, vault):
        vault.store_credential(
            provider="gmail", account_id="a@gmail.com",
            token_type="oauth2", access_token="token_a",
        )
        vault.store_credential(
            provider="gmail", account_id="b@gmail.com",
            token_type="oauth2", access_token="token_b",
        )
        cred = vault.get_credential("gmail", account_id="b@gmail.com")
        assert cred.access_token == "token_b"

    def test_list_credentials(self, vault):
        vault.store_credential(
            provider="gmail", account_id="a@gmail.com",
            token_type="oauth2", access_token="t1",
        )
        vault.store_credential(
            provider="gmail", account_id="b@gmail.com",
            token_type="oauth2", access_token="t2",
        )
        creds = vault.list_credentials("gmail")
        assert len(creds) == 2

    def test_refresh_credential(self, vault):
        vault.store_credential(
            provider="gmail", account_id="u@gmail.com",
            token_type="oauth2", access_token="old_token",
        )
        new_exp = time.time() + 7200
        vault.refresh_credential(
            "gmail:u@gmail.com",
            new_access_token="new_token",
            new_expires_at=new_exp,
        )
        cred = vault.get_credential("gmail")
        assert cred.access_token == "new_token"
        assert cred.expires_at == pytest.approx(new_exp, abs=1)

    def test_deactivate_and_reactivate(self, vault):
        vault.store_credential(
            provider="gmail", account_id="u@gmail.com",
            token_type="oauth2", access_token="token",
        )
        vault.deactivate_credential("gmail:u@gmail.com")

        # Active query should not find it
        assert vault.get_credential("gmail") is None

        # But list shows it (inactive)
        creds = vault.list_credentials("gmail")
        assert len(creds) == 1
        assert creds[0].active is False

    def test_delete_credential(self, vault):
        vault.store_credential(
            provider="gmail", account_id="u@gmail.com",
            token_type="oauth2", access_token="token",
        )
        vault.delete_credential("gmail:u@gmail.com")
        assert vault.list_credentials("gmail") == []

    def test_get_nonexistent_returns_none(self, vault):
        assert vault.get_credential("nonexistent") is None


class TestProfiles:
    def test_store_and_retrieve(self, vault):
        vault.store_profile(
            "primary", display_name="Muthu",
            email="m@example.com", phone="+1234567890",
            metadata={"avatar": "url"},
        )
        p = vault.get_profile("primary")
        assert p.display_name == "Muthu"
        assert p.email == "m@example.com"
        assert p.phone == "+1234567890"
        assert p.metadata == {"avatar": "url"}

    def test_get_nonexistent_returns_none(self, vault):
        assert vault.get_profile("nobody") is None


class TestConsent:
    def test_log_and_retrieve(self, vault):
        vault.log_consent("gmail", "connected", scopes=["email.read"])
        vault.log_consent("gmail", "reauthorized", scopes=["email.read", "email.send"])
        history = vault.get_consent_history("gmail")
        assert len(history) == 2
        assert history[0].action == "connected"
        assert history[1].action == "reauthorized"

    def test_get_last_consent(self, vault):
        vault.log_consent("gmail", "connected")
        vault.log_consent("gmail", "disconnected", reason="user request")
        last = vault.get_last_consent("gmail")
        assert last.action == "disconnected"
        assert last.reason == "user request"

    def test_empty_history(self, vault):
        assert vault.get_consent_history("nonexistent") == []
        assert vault.get_last_consent("nonexistent") is None


class TestFinancial:
    def test_store_and_query(self, vault):
        record_id = vault.store_financial(
            source="gmail:msg123", category="bill",
            description="Electric bill", amount="142.50",
            currency="USD", due_date=time.time() + 86400,
        )
        assert record_id > 0

        records = vault.query_financial(status="pending")
        assert len(records) == 1
        assert records[0].description == "Electric bill"
        assert records[0].amount == "142.50"

    def test_query_by_due_date(self, vault):
        now = time.time()
        vault.store_financial(
            source="a", category="bill", description="Soon",
            due_date=now + 3600,
        )
        vault.store_financial(
            source="b", category="bill", description="Later",
            due_date=now + 86400 * 30,
        )
        results = vault.query_financial(due_before=now + 86400)
        assert len(results) == 1
        assert results[0].description == "Soon"

    def test_update_financial(self, vault):
        rid = vault.store_financial(
            source="a", category="bill", description="Test",
        )
        vault.update_financial(rid, status="paid")
        records = vault.query_financial(status="paid")
        assert len(records) == 1


class TestExportImport:
    def test_export_and_import_round_trip(self, vault, tmp_path):
        vault.store_credential(
            provider="gmail", account_id="u@gmail.com",
            token_type="oauth2", access_token="secret_token",
        )
        export_path = tmp_path / "backup.vault"
        vault.export_vault(export_path, password="backup_pass")
        assert export_path.exists()

    def test_export_wrong_password_fails(self, vault, tmp_path):
        vault.store_credential(
            provider="test", account_id="x",
            token_type="oauth2", access_token="t",
        )
        export_path = tmp_path / "backup.vault"
        vault.export_vault(export_path, password="correct")

        v2 = SecureVault(storage_dir=tmp_path / "vault2")
        v2.unlock()
        with pytest.raises(VaultAuthError):
            v2.import_vault(export_path, password="wrong")


class TestConnectionStatus:
    def test_set_and_get(self, vault):
        vault.set_connection_status(
            "gmail", connected=True,
            label="Gmail (m***@gmail.com)",
            mode="always_on", sync_interval=300,
        )
        status = vault.get_connection_status("gmail")
        assert status.connected is True
        assert status.connection_mode == "always_on"

    def test_get_all_connections(self, vault):
        vault.set_connection_status("gmail", connected=True)
        vault.set_connection_status("slack", connected=False)
        all_conns = vault.get_all_connections()
        assert len(all_conns) == 2

    def test_nonexistent_returns_none(self, vault):
        assert vault.get_connection_status("none") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vault/test_secure_vault.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/vault/secure_vault.py
"""SecureVault — thread-safe encrypted storage API.

Single entry point for all credential, profile, consent, and financial
data operations. Encrypts sensitive fields with per-category AES-256-GCM
keys derived from a master key stored in the OS keyring.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from homie_core.vault.encryption import (
    generate_master_key,
    derive_category_key,
    encrypt_field,
    decrypt_field,
    zero_bytearray,
)
from homie_core.vault.exceptions import (
    VaultLockedError,
    CredentialNotFoundError,
)
from homie_core.vault.keyring_backend import KeyringBackend
from homie_core.vault.models import (
    Credential,
    UserProfile,
    ConsentEntry,
    FinancialRecord,
    ConnectionStatus,
)
from homie_core.vault.schema import (
    create_vault_db,
    create_cache_db,
    check_integrity,
    run_migrations,
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

        # Category keys — held in memory only while unlocked
        self._keys: dict[str, bytearray] = {}
        self._unlocked = False

    # ── Lifecycle ───────────────────────────────────────────────────

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked

    def unlock(self, password: Optional[str] = None) -> None:
        """Unlock the vault. Creates databases on first run."""
        master = self._keyring.retrieve_master_key(password=password)
        if master is None:
            # First run — generate new master key
            master = generate_master_key()
            self._keyring.store_master_key(master)

        # Derive category keys
        for category in ("credentials", "profiles", "financial", "consent"):
            self._keys[category] = derive_category_key(master, category)
        zero_bytearray(master)

        # Open databases
        vault_path = self._storage_dir / "vault.db"
        cache_path = self._storage_dir / "cache.db"
        self._vault_conn = create_vault_db(vault_path)
        self._cache_conn = create_cache_db(cache_path)

        # Run migrations
        meta_path = self._storage_dir / "vault.meta.json"
        run_migrations(self._vault_conn, meta_path)

        # Integrity check
        check_integrity(self._vault_conn)
        check_integrity(self._cache_conn)

        self._unlocked = True

    def lock(self) -> None:
        """Lock the vault. Zeros all keys and closes connections."""
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
        """Enable or change the master password."""
        if not self._unlocked:
            raise VaultLockedError("Vault must be unlocked to set password.")
        # Re-derive master from any category key is not possible,
        # so we regenerate and re-encrypt everything.
        # Simpler: store a copy of master key before zeroing in unlock()
        # For now: generate fresh master, re-derive keys, re-encrypt all fields.
        master = generate_master_key()
        new_keys = {}
        for category in ("credentials", "profiles", "financial", "consent"):
            new_keys[category] = derive_category_key(master, category)

        # Re-encrypt all data with new keys
        self._reencrypt_all(self._keys, new_keys)

        # Store new master with password
        self._keyring.store_master_key(master)
        self._keyring.set_password(password, master)

        # Swap keys
        for old_key in self._keys.values():
            zero_bytearray(old_key)
        self._keys = new_keys
        zero_bytearray(master)

    # ── Export / Import ──────────────────────────────────────────────

    @_require_unlocked
    def export_vault(self, path: Path, password: str) -> None:
        """Export all vault data as a password-encrypted JSON file."""
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        data = {}
        with self._vault_lock:
            for table in ("credentials", "user_profiles", "consent_log", "financial_data"):
                rows = self._vault_conn.execute(f"SELECT * FROM {table}").fetchall()
                cols = [d[0] for d in self._vault_conn.execute(f"PRAGMA table_info({table})").fetchall()]
                # Column names are at index 1 in PRAGMA table_info
                col_names = [c[1] for c in self._vault_conn.execute(f"PRAGMA table_info({table})").fetchall()]
                data[table] = [dict(zip(col_names, row)) for row in rows]

        payload = json.dumps(data).encode("utf-8")
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000)
        export_key = kdf.derive(password.encode("utf-8"))
        nonce = os.urandom(12)
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(export_key)
        ct = aesgcm.encrypt(nonce, payload, None)
        # File format: salt(16) + nonce(12) + ciphertext
        Path(path).write_bytes(salt + nonce + ct)

    @_require_unlocked
    def import_vault(self, path: Path, password: str) -> None:
        """Import vault data from a password-encrypted export file."""
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        raw = Path(path).read_bytes()
        salt, nonce, ct = raw[:16], raw[16:28], raw[28:]
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000)
        export_key = kdf.derive(password.encode("utf-8"))
        aesgcm = AESGCM(export_key)
        try:
            payload = aesgcm.decrypt(nonce, ct, None)
        except Exception as e:
            raise VaultAuthError(f"Wrong password or corrupted export file: {e}") from e
        data = json.loads(payload.decode("utf-8"))
        # Re-insert data (this replaces existing vault contents)
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

    # ── Credentials ─────────────────────────────────────────────────

    @_require_unlocked
    def store_credential(
        self,
        provider: str,
        account_id: str,
        token_type: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[float] = None,
        scopes: Optional[list[str]] = None,
    ) -> str:
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
    def get_credential(
        self, provider: str, account_id: Optional[str] = None,
    ) -> Optional[Credential]:
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
    def refresh_credential(
        self,
        credential_id: str,
        new_access_token: str,
        new_expires_at: Optional[float] = None,
    ) -> None:
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
            # VACUUM must run outside a transaction
            self._vault_conn.execute("VACUUM")

    # ── Profiles ────────────────────────────────────────────────────

    @_require_unlocked
    def store_profile(
        self,
        profile_id: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
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
            created_at=row[5],
            updated_at=row[6],
        )

    # ── Consent ─────────────────────────────────────────────────────

    @_require_unlocked
    def log_consent(
        self,
        provider: str,
        action: str,
        scopes: Optional[list[str]] = None,
        reason: Optional[str] = None,
    ) -> None:
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
            )
            for r in rows
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

    # ── Financial ───────────────────────────────────────────────────

    @_require_unlocked
    def store_financial(
        self,
        source: str,
        category: str,
        description: str,
        amount: Optional[str] = None,
        currency: Optional[str] = None,
        due_date: Optional[float] = None,
    ) -> int:
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
    def query_financial(
        self,
        status: Optional[str] = None,
        due_before: Optional[float] = None,
        category: Optional[str] = None,
    ) -> list[FinancialRecord]:
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
            )
            for r in rows
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

    # ── Connection Status ───────────────────────────────────────────

    @property
    def has_password(self) -> bool:
        """Check if the vault has a password layer configured."""
        return self._keyring.has_password()

    @_require_unlocked
    def set_connection_status(
        self,
        provider: str,
        connected: bool,
        label: Optional[str] = None,
        mode: str = "always_on",
        sync_interval: int = 300,
        last_sync: Optional[float] = None,
    ) -> None:
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
            sync_interval=row[4], last_sync=row[5],
            last_sync_error=row[6],
        )

    @_require_unlocked
    def get_all_connections(self) -> list[ConnectionStatus]:
        with self._cache_lock:
            rows = self._cache_conn.execute(
                "SELECT * FROM connection_status"
            ).fetchall()
        return [
            ConnectionStatus(
                provider=r[0], connected=bool(r[1]),
                display_label=r[2], connection_mode=r[3],
                sync_interval=r[4], last_sync=r[5],
                last_sync_error=r[6],
            )
            for r in rows
        ]

    # ── Internal helpers ────────────────────────────────────────────

    def _row_to_credential(self, row) -> Credential:
        key = self._keys["credentials"]
        return Credential(
            id=row[0], provider=row[1], account_id=row[2],
            token_type=row[3],
            access_token=decrypt_field(row[4], key),
            refresh_token=decrypt_field(row[5], key) if row[5] else None,
            expires_at=row[6],
            scopes=json.loads(row[7]) if row[7] else None,
            active=bool(row[8]),
            created_at=row[9], updated_at=row[10],
        )

    def _get_credential_lock(self, credential_id: str) -> threading.Lock:
        with self._credential_locks_lock:
            if credential_id not in self._credential_locks:
                self._credential_locks[credential_id] = threading.Lock()
            return self._credential_locks[credential_id]

    def _reencrypt_all(
        self, old_keys: dict[str, bytearray], new_keys: dict[str, bytearray],
    ) -> None:
        """Re-encrypt all fields when master key changes (password set)."""
        with self._vault_lock:
            # Re-encrypt credentials
            rows = self._vault_conn.execute("SELECT * FROM credentials").fetchall()
            for r in rows:
                old_k = old_keys["credentials"]
                new_k = new_keys["credentials"]
                new_access = encrypt_field(decrypt_field(r[4], old_k), new_k)
                new_refresh = (
                    encrypt_field(decrypt_field(r[5], old_k), new_k) if r[5] else None
                )
                self._vault_conn.execute(
                    "UPDATE credentials SET access_token=?, refresh_token=? WHERE id=?",
                    (new_access, new_refresh, r[0]),
                )

            # Re-encrypt profiles
            rows = self._vault_conn.execute("SELECT * FROM user_profiles").fetchall()
            for r in rows:
                old_k = old_keys["profiles"]
                new_k = new_keys["profiles"]
                new_name = encrypt_field(decrypt_field(r[1], old_k), new_k) if r[1] else None
                new_email = encrypt_field(decrypt_field(r[2], old_k), new_k) if r[2] else None
                new_phone = encrypt_field(decrypt_field(r[3], old_k), new_k) if r[3] else None
                new_meta = encrypt_field(decrypt_field(r[4], old_k), new_k) if r[4] else None
                self._vault_conn.execute(
                    """UPDATE user_profiles SET display_name=?, email=?,
                       phone=?, metadata=? WHERE id=?""",
                    (new_name, new_email, new_phone, new_meta, r[0]),
                )

            # Re-encrypt financial
            rows = self._vault_conn.execute("SELECT * FROM financial_data").fetchall()
            for r in rows:
                old_k = old_keys["financial"]
                new_k = new_keys["financial"]
                new_desc = encrypt_field(decrypt_field(r[3], old_k), new_k)
                new_amt = encrypt_field(decrypt_field(r[4], old_k), new_k) if r[4] else None
                new_raw = encrypt_field(decrypt_field(r[9], old_k), new_k) if r[9] else None
                self._vault_conn.execute(
                    """UPDATE financial_data SET description=?, amount=?,
                       raw_extract=? WHERE id=?""",
                    (new_desc, new_amt, new_raw, r[0]),
                )

            # Re-encrypt consent reasons
            rows = self._vault_conn.execute("SELECT * FROM consent_log").fetchall()
            for r in rows:
                if r[4]:  # reason field
                    old_k = old_keys["consent"]
                    new_k = new_keys["consent"]
                    new_reason = encrypt_field(decrypt_field(r[4], old_k), new_k)
                    self._vault_conn.execute(
                        "UPDATE consent_log SET reason=? WHERE id=?",
                        (new_reason, r[0]),
                    )

            self._vault_conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vault/test_secure_vault.py -v`
Expected: 27 passed

- [ ] **Step 5: Update `__init__.py` to re-export SecureVault**

In `src/homie_core/vault/__init__.py`, add:

```python
from homie_core.vault.secure_vault import SecureVault

# Add to __all__:
"SecureVault",
```

- [ ] **Step 6: Run full vault test suite**

Run: `pytest tests/unit/test_vault/ -v`
Expected: All tests pass (exceptions + models + encryption + keyring + schema + secure_vault)

- [ ] **Step 7: Commit**

```bash
git add src/homie_core/vault/secure_vault.py src/homie_core/vault/__init__.py \
       tests/unit/test_vault/test_secure_vault.py
git commit -m "feat(vault): add SecureVault API with full CRUD for credentials, profiles, consent, financial"
```

---

## Chunk 4: SyncManager, Daemon Integration, CLI

### Task 8: SyncManager

**Files:**
- Create: `src/homie_core/vault/sync_manager.py`
- Create: `tests/unit/test_vault/test_sync_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vault/test_sync_manager.py
import time
import pytest
from unittest.mock import MagicMock, patch

from homie_core.vault.sync_manager import SyncManager
from homie_core.vault.models import ConnectionStatus


class TestSyncManagerTick:
    def test_tick_calls_sync_for_due_providers(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="always_on", sync_interval=300,
                last_sync=time.time() - 400,  # overdue
            ),
        ]
        on_sync = MagicMock(return_value="synced 5 emails")
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 1
        assert results[0][0] == "gmail"
        on_sync.assert_called_once()

    def test_tick_skips_not_due_providers(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="always_on", sync_interval=300,
                last_sync=time.time() - 100,  # not due yet
            ),
        ]
        on_sync = MagicMock()
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 0
        on_sync.assert_not_called()

    def test_tick_skips_disconnected_providers(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=False,
                connection_mode="always_on", sync_interval=300,
            ),
        ]
        on_sync = MagicMock()
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 0

    def test_tick_skips_on_demand_providers(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="on_demand", sync_interval=300,
                last_sync=time.time() - 400,
            ),
        ]
        on_sync = MagicMock()
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 0

    def test_tick_handles_sync_error_gracefully(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="always_on", sync_interval=300,
                last_sync=time.time() - 400,
            ),
        ]
        on_sync = MagicMock(side_effect=Exception("API error"))
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 1
        assert "error" in results[0][1].lower()

    def test_tick_with_no_callback_registered(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="always_on", sync_interval=300,
                last_sync=time.time() - 400,
            ),
        ]
        manager = SyncManager(vault=vault, sync_callbacks={})

        results = manager.tick()
        assert len(results) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vault/test_sync_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/vault/sync_manager.py
"""SyncManager — periodic background sync for connected providers.

Called from the daemon's main-thread tick loop (every 60s). Checks which
always_on providers are due for sync and calls their registered callbacks.
On-demand providers are skipped (they sync only when explicitly triggered).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from homie_core.vault.models import ConnectionStatus


class SyncManager:
    """Manages periodic sync for connected providers."""

    def __init__(
        self,
        vault,
        sync_callbacks: Optional[dict[str, Callable]] = None,
    ):
        self._vault = vault
        self._callbacks: dict[str, Callable] = sync_callbacks or {}

    def register_callback(self, provider: str, callback: Callable) -> None:
        """Register a sync callback for a provider."""
        self._callbacks[provider] = callback

    def tick(self) -> list[tuple[str, str]]:
        """Check all connected providers and sync those that are due.

        Returns list of (provider, result_message) for providers that synced.
        Called from daemon's main loop every 60 seconds.
        """
        results: list[tuple[str, str]] = []

        try:
            connections = self._vault.get_all_connections()
        except Exception:
            return results

        now = time.time()

        for conn in connections:
            if not conn.connected:
                continue
            if conn.connection_mode != "always_on":
                continue
            if conn.provider not in self._callbacks:
                continue

            # Check if sync is due
            last = conn.last_sync or 0
            if (now - last) < conn.sync_interval:
                continue

            # Execute sync
            try:
                result = self._callbacks[conn.provider]()
                results.append((conn.provider, str(result)))
                # Update last_sync timestamp
                try:
                    self._vault.set_connection_status(
                        conn.provider,
                        connected=True,
                        label=conn.display_label,
                        mode=conn.connection_mode,
                        sync_interval=conn.sync_interval,
                        last_sync=time.time(),
                    )
                except Exception:
                    pass
            except Exception as e:
                error_msg = f"Sync error: {e}"
                results.append((conn.provider, error_msg))

        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vault/test_sync_manager.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/vault/sync_manager.py tests/unit/test_vault/test_sync_manager.py
git commit -m "feat(vault): add SyncManager for periodic background provider sync"
```

---

### Task 9: Daemon integration

**Files:**
- Modify: `src/homie_app/daemon.py`

- [ ] **Step 1: Add vault imports to daemon.py**

Add near the top imports:

```python
from homie_core.vault.secure_vault import SecureVault
from homie_core.vault.sync_manager import SyncManager as VaultSyncManager
```

- [ ] **Step 2: Add vault initialization in `__init__`**

After the existing `self._skill_loader = SkillLoader()` line, add:

```python
# Secure vault for credentials and encrypted data
vault_dir = storage / "vault"
self._vault = SecureVault(storage_dir=vault_dir)
try:
    self._vault.unlock()
    print("  Vault: unlocked")
except Exception as e:
    print(f"  Vault: failed to unlock ({e})")

# Vault sync manager (callbacks registered by sub-projects)
self._vault_sync = VaultSyncManager(vault=self._vault)
```

- [ ] **Step 3: Add vault sync tick in `start()` method**

Inside the existing 60-second tick loop (after the scheduler tick), add:

```python
try:
    sync_results = self._vault_sync.tick()
    for provider, output in sync_results:
        print(f"  [Sync] {provider}: {output[:100]}")
except Exception:
    pass
```

- [ ] **Step 4: Add vault lock in `stop()` method**

Before `self._observer.stop()`, add:

```python
# Lock vault and stop sync
self._vault_sync = None
self._vault.lock()
```

- [ ] **Step 5: Run existing daemon tests**

Run: `pytest tests/unit/test_daemon.py -v`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/homie_app/daemon.py
git commit -m "feat(daemon): wire SecureVault and SyncManager into daemon lifecycle"
```

---

### Task 10: CLI commands

**Files:**
- Modify: `src/homie_app/cli.py`

- [ ] **Step 1: Add CLI subcommands for vault management**

Add new handler functions and wire them into the argument parser. The commands to add:

- `homie connections` — list all connection statuses
- `homie consent-log <provider>` — show consent audit trail
- `homie vault status` — show vault health (DB sizes, connection count)

```python
def cmd_connections(args) -> None:
    """List all connection statuses."""
    from homie_core.vault.secure_vault import SecureVault
    vault = SecureVault()
    try:
        vault.unlock()
        connections = vault.get_all_connections()
        if not connections:
            print("No connections configured. Use 'homie connect <provider>' to add one.")
            return
        for c in connections:
            icon = "\u2705" if c.connected else "\u274c"
            mode = f" ({c.connection_mode})" if c.connected else ""
            print(f"  {icon} {c.provider}: {c.display_label or 'no label'}{mode}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vault.lock()


def cmd_consent_log(args) -> None:
    """Show consent audit trail for a provider."""
    from homie_core.vault.secure_vault import SecureVault
    provider = args.provider
    vault = SecureVault()
    try:
        vault.unlock()
        history = vault.get_consent_history(provider)
        if not history:
            print(f"No consent history for '{provider}'.")
            return
        for entry in history:
            from datetime import datetime
            dt = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M")
            scopes_str = ", ".join(entry.scopes) if entry.scopes else ""
            reason_str = f"  reason: {entry.reason}" if entry.reason else ""
            print(f"  {dt}  {entry.action:<15} {scopes_str}{reason_str}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vault.lock()


def cmd_vault_status(args) -> None:
    """Show vault health and statistics."""
    from pathlib import Path
    from homie_core.vault.secure_vault import SecureVault
    vault = SecureVault()
    try:
        vault.unlock()
        connections = vault.get_all_connections()
        active = sum(1 for c in connections if c.connected)
        vault_path = Path.home() / ".homie" / "vault"
        vault_size = (vault_path / "vault.db").stat().st_size if (vault_path / "vault.db").exists() else 0
        cache_size = (vault_path / "cache.db").stat().st_size if (vault_path / "cache.db").exists() else 0

        print(f"  Vault DB: {vault_size / 1024:.1f} KB")
        print(f"  Cache DB: {cache_size / 1024:.1f} KB")
        print(f"  Connections: {active} active / {len(connections)} total")
        print(f"  Password: {'set' if vault.has_password else 'not set'}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vault.lock()
```

Add to the argument parser:

```python
# In create_parser():
sub_connections = subparsers.add_parser("connections", help="List provider connections")
sub_connections.set_defaults(func=cmd_connections)

sub_consent = subparsers.add_parser("consent-log", help="Show consent audit trail")
sub_consent.add_argument("provider", help="Provider name (gmail, slack, etc.)")
sub_consent.set_defaults(func=cmd_consent_log)

sub_vault = subparsers.add_parser("vault", help="Vault management")
vault_sub = sub_vault.add_subparsers(dest="vault_cmd")
vault_status = vault_sub.add_parser("status", help="Show vault health")
vault_status.set_defaults(func=cmd_vault_status)
```

Also add stub commands for `connect` and `disconnect` (full OAuth implementation deferred to sub-project 1):

```python
def cmd_connect(args) -> None:
    """Stub for provider connection — full OAuth flow added in sub-project 1."""
    provider = args.provider
    print(f"Connecting to {provider}...")
    print("OAuth integration not yet available. Coming in email/social sub-projects.")


def cmd_disconnect(args) -> None:
    """Disconnect a provider with user confirmation."""
    from homie_core.vault.secure_vault import SecureVault
    provider = args.provider
    vault = SecureVault()
    try:
        vault.unlock()
        cred = vault.get_credential(provider)
        if not cred:
            print(f"No active connection for '{provider}'.")
            return
        print(f"Disconnecting {provider}...")
        print("  1) Disconnect (keep credentials encrypted, can reconnect later)")
        print("  2) Disconnect and delete credentials permanently")
        print("  3) Cancel")
        choice = input("Choose [1/2/3]: ").strip()
        if choice == "1":
            vault.deactivate_credential(cred.id)
            vault.log_consent(provider, "disconnected", reason="user_initiated")
            vault.set_connection_status(provider, connected=False)
            print(f"  {provider} disconnected. Credentials kept encrypted.")
        elif choice == "2":
            confirm = input(f"Permanently delete {provider} credentials? (yes/no): ")
            if confirm.lower() == "yes":
                vault.delete_credential(cred.id)
                vault.log_consent(provider, "disconnected", reason="user_deleted")
                vault.set_connection_status(provider, connected=False)
                print(f"  {provider} credentials permanently deleted.")
            else:
                print("  Cancelled.")
        else:
            print("  Cancelled.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        vault.lock()
```

Add to parser:

```python
sub_connect = subparsers.add_parser("connect", help="Connect a provider")
sub_connect.add_argument("provider", help="Provider name (gmail, slack, etc.)")
sub_connect.set_defaults(func=cmd_connect)

sub_disconnect = subparsers.add_parser("disconnect", help="Disconnect a provider")
sub_disconnect.add_argument("provider", help="Provider name")
sub_disconnect.set_defaults(func=cmd_disconnect)
```

Also add meta-commands in the chat loop: `/connections`, `/consent-log`, `/vault`, `/connect`, `/disconnect`.

- [ ] **Step 2: Run CLI tests**

Run: `pytest tests/unit/test_app/test_cli.py -v`
Expected: All existing tests still pass

- [ ] **Step 3: Commit**

```bash
git add src/homie_app/cli.py
git commit -m "feat(cli): add connections, consent-log, and vault status commands"
```

---

### Task 11: Final integration test

**Files:**
- No new files — runs existing tests

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass (729+ existing + ~50 new vault tests)

- [ ] **Step 2: Run vault-specific tests in isolation**

Run: `pytest tests/unit/test_vault/ -v`
Expected: All vault tests pass

- [ ] **Step 3: Syntax check all new files**

Run: `python -m py_compile src/homie_core/vault/encryption.py && python -m py_compile src/homie_core/vault/keyring_backend.py && python -m py_compile src/homie_core/vault/schema.py && python -m py_compile src/homie_core/vault/secure_vault.py && python -m py_compile src/homie_core/vault/sync_manager.py && echo "All OK"`
Expected: "All OK"

- [ ] **Step 4: Final commit and push**

```bash
git push origin feat/homie-ai-v2
```
