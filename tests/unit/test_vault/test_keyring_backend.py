import os
import pytest
from unittest.mock import patch

from homie_core.vault.keyring_backend import KeyringBackend
from homie_core.vault.exceptions import VaultAuthError, RateLimitError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backend(tmp_path: pytest.TempPathFactory, *, use_keyring: bool = False) -> KeyringBackend:
    """Create a KeyringBackend that always uses the file fallback."""
    with patch("homie_core.vault.keyring_backend._keyring_available", return_value=use_keyring):
        return KeyringBackend(fallback_dir=tmp_path)


# ---------------------------------------------------------------------------
# Store / retrieve
# ---------------------------------------------------------------------------

class TestKeyringBackendStore:
    def test_store_and_retrieve_master_key(self, tmp_path):
        backend = _backend(tmp_path)
        key = bytearray(os.urandom(32))
        backend.store_master_key(key)
        retrieved = backend.retrieve_master_key()
        assert retrieved == key

    def test_retrieve_returns_bytearray(self, tmp_path):
        backend = _backend(tmp_path)
        key = bytearray(os.urandom(32))
        backend.store_master_key(key)
        retrieved = backend.retrieve_master_key()
        assert isinstance(retrieved, bytearray)

    def test_retrieve_nonexistent_returns_none(self, tmp_path):
        backend = _backend(tmp_path)
        assert backend.retrieve_master_key() is None


# ---------------------------------------------------------------------------
# Password layer
# ---------------------------------------------------------------------------

class TestPasswordLayer:
    def test_set_and_unlock_with_password(self, tmp_path):
        backend = _backend(tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        backend.set_password("my-secret-pass", master)
        retrieved = backend.retrieve_master_key(password="my-secret-pass")
        assert retrieved == master

    def test_wrong_password_raises_auth_error(self, tmp_path):
        backend = _backend(tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        backend.set_password("correct-pass", master)
        with pytest.raises(VaultAuthError):
            backend.retrieve_master_key(password="wrong-pass")

    def test_has_password(self, tmp_path):
        backend = _backend(tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        assert backend.has_password() is False
        backend.set_password("pass", master)
        assert backend.has_password() is True


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_rate_limit_after_5_failures(self, tmp_path):
        backend = _backend(tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        backend.set_password("correct", master)
        for i in range(5):
            with pytest.raises(VaultAuthError):
                backend.retrieve_master_key(password="wrong")
        with pytest.raises(RateLimitError):
            backend.retrieve_master_key(password="wrong")

    def test_successful_unlock_resets_counter(self, tmp_path):
        backend = _backend(tmp_path)
        master = bytearray(os.urandom(32))
        backend.store_master_key(master)
        backend.set_password("correct", master)
        for _ in range(3):
            with pytest.raises(VaultAuthError):
                backend.retrieve_master_key(password="wrong")
        result = backend.retrieve_master_key(password="correct")
        assert result == master
        for _ in range(5):
            with pytest.raises(VaultAuthError):
                backend.retrieve_master_key(password="wrong")
        with pytest.raises(RateLimitError):
            backend.retrieve_master_key(password="wrong")


# ---------------------------------------------------------------------------
# File fallback
# ---------------------------------------------------------------------------

class TestFileFallback:
    def test_uses_file_when_keyring_unavailable(self, tmp_path):
        with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
            backend = KeyringBackend(fallback_dir=tmp_path)
            key = bytearray(os.urandom(32))
            backend.store_master_key(key)
            keyfile = tmp_path / ".keyfile"
            assert keyfile.exists()
            retrieved = backend.retrieve_master_key()
            assert retrieved == key

    def test_delete_master_key(self, tmp_path):
        backend = _backend(tmp_path)
        key = bytearray(os.urandom(32))
        backend.store_master_key(key)
        backend.delete_master_key()
        assert backend.retrieve_master_key() is None
