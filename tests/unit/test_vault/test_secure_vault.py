import time
import pytest
from unittest.mock import patch

from homie_core.vault.secure_vault import SecureVault
from homie_core.vault.exceptions import VaultLockedError, VaultAuthError


@pytest.fixture
def vault(tmp_path):
    with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
        v = SecureVault(storage_dir=tmp_path / "vault")
        v.unlock()
        yield v
        if v.is_unlocked:
            v.lock()


@pytest.fixture
def locked_vault(tmp_path):
    with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
        yield SecureVault(storage_dir=tmp_path / "vault")


class TestVaultLifecycle:
    def test_unlock_creates_databases(self, tmp_path):
        with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
            v = SecureVault(storage_dir=tmp_path / "vault")
            v.unlock()
            assert (tmp_path / "vault" / "vault.db").exists()
            assert (tmp_path / "vault" / "cache.db").exists()
            v.lock()

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
        with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
            v = SecureVault(storage_dir=tmp_path / "vault")
            v.unlock()
            v.set_password("secret123")
            v.lock()
            v2 = SecureVault(storage_dir=tmp_path / "vault")
            v2.unlock(password="secret123")
            assert v2.is_unlocked is True
            v2.lock()

    def test_wrong_password_raises(self, tmp_path):
        with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
            v = SecureVault(storage_dir=tmp_path / "vault")
            v.unlock()
            v.set_password("correct")
            v.lock()
            v2 = SecureVault(storage_dir=tmp_path / "vault")
            with pytest.raises(VaultAuthError):
                v2.unlock(password="wrong")

    def test_set_password_preserves_data(self, tmp_path):
        with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
            v = SecureVault(storage_dir=tmp_path / "vault")
            v.unlock()
            v.store_credential(provider="gmail", account_id="u@gmail.com",
                              token_type="oauth2", access_token="my_secret_token",
                              refresh_token="my_refresh")
            v.store_profile("primary", display_name="Test User", email="u@gmail.com")
            v.log_consent("gmail", "connected", reason="user approved")
            v.store_financial(source="a", category="bill", description="Electric", amount="99.50")
            v.set_password("new_pass")
            v.lock()
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
        with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
            v = SecureVault(storage_dir=tmp_path / "vault")
            v.unlock()
            assert v.has_password is False
            v.set_password("pass")
            assert v.has_password is True
            v.lock()


class TestCredentials:
    def test_store_and_retrieve(self, vault):
        cred_id = vault.store_credential(
            provider="gmail", account_id="user@gmail.com",
            token_type="oauth2", access_token="access_123",
            refresh_token="refresh_456",
            expires_at=time.time() + 3600, scopes=["email.read"],
        )
        assert cred_id == "gmail:user@gmail.com"
        cred = vault.get_credential("gmail")
        assert cred.access_token == "access_123"
        assert cred.refresh_token == "refresh_456"
        assert cred.scopes == ["email.read"]

    def test_get_by_account_id(self, vault):
        vault.store_credential(provider="gmail", account_id="a@gmail.com",
                              token_type="oauth2", access_token="token_a")
        vault.store_credential(provider="gmail", account_id="b@gmail.com",
                              token_type="oauth2", access_token="token_b")
        cred = vault.get_credential("gmail", account_id="b@gmail.com")
        assert cred.access_token == "token_b"

    def test_list_credentials(self, vault):
        vault.store_credential(provider="gmail", account_id="a@gmail.com",
                              token_type="oauth2", access_token="t1")
        vault.store_credential(provider="gmail", account_id="b@gmail.com",
                              token_type="oauth2", access_token="t2")
        creds = vault.list_credentials("gmail")
        assert len(creds) == 2

    def test_refresh_credential(self, vault):
        vault.store_credential(provider="gmail", account_id="u@gmail.com",
                              token_type="oauth2", access_token="old_token")
        new_exp = time.time() + 7200
        vault.refresh_credential("gmail:u@gmail.com",
                                new_access_token="new_token", new_expires_at=new_exp)
        cred = vault.get_credential("gmail")
        assert cred.access_token == "new_token"
        assert cred.expires_at == pytest.approx(new_exp, abs=1)

    def test_deactivate_and_reactivate(self, vault):
        vault.store_credential(provider="gmail", account_id="u@gmail.com",
                              token_type="oauth2", access_token="token")
        vault.deactivate_credential("gmail:u@gmail.com")
        assert vault.get_credential("gmail") is None
        creds = vault.list_credentials("gmail")
        assert len(creds) == 1
        assert creds[0].active is False

    def test_delete_credential(self, vault):
        vault.store_credential(provider="gmail", account_id="u@gmail.com",
                              token_type="oauth2", access_token="token")
        vault.delete_credential("gmail:u@gmail.com")
        assert vault.list_credentials("gmail") == []

    def test_get_nonexistent_returns_none(self, vault):
        assert vault.get_credential("nonexistent") is None


class TestProfiles:
    def test_store_and_retrieve(self, vault):
        vault.store_profile("primary", display_name="Muthu",
                           email="m@example.com", phone="+1234567890",
                           metadata={"avatar": "url"})
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
        vault.store_financial(source="a", category="bill", description="Soon",
                            due_date=now + 3600)
        vault.store_financial(source="b", category="bill", description="Later",
                            due_date=now + 86400 * 30)
        results = vault.query_financial(due_before=now + 86400)
        assert len(results) == 1
        assert results[0].description == "Soon"

    def test_update_financial(self, vault):
        rid = vault.store_financial(source="a", category="bill", description="Test")
        vault.update_financial(rid, status="paid")
        records = vault.query_financial(status="paid")
        assert len(records) == 1


class TestExportImport:
    def test_export_creates_file(self, vault, tmp_path):
        vault.store_credential(provider="gmail", account_id="u@gmail.com",
                              token_type="oauth2", access_token="secret_token")
        export_path = tmp_path / "backup.vault"
        vault.export_vault(export_path, password="backup_pass")
        assert export_path.exists()

    def test_export_wrong_password_fails(self, vault, tmp_path):
        vault.store_credential(provider="test", account_id="x",
                              token_type="oauth2", access_token="t")
        export_path = tmp_path / "backup.vault"
        vault.export_vault(export_path, password="correct")
        with patch("homie_core.vault.keyring_backend._keyring_available", return_value=False):
            v2 = SecureVault(storage_dir=tmp_path / "vault2")
            v2.unlock()
            with pytest.raises(VaultAuthError):
                v2.import_vault(export_path, password="wrong")
            v2.lock()


class TestConnectionStatus:
    def test_set_and_get(self, vault):
        vault.set_connection_status("gmail", connected=True,
                                   label="Gmail (m***@gmail.com)",
                                   mode="always_on", sync_interval=300)
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

    def test_last_sync_is_stored(self, vault):
        now = time.time()
        vault.set_connection_status("gmail", connected=True, last_sync=now)
        status = vault.get_connection_status("gmail")
        assert status.last_sync == pytest.approx(now, abs=1)
