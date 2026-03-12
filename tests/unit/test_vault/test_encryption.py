import os
import pytest

from homie_core.vault.encryption import (
    generate_master_key, derive_category_key,
    encrypt_field, decrypt_field, zero_bytearray,
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
        assert ct1 != ct2

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
