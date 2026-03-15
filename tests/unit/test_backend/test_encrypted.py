"""Tests for EncryptedVaultBackend."""
from __future__ import annotations

import pytest

from homie_core.backend.encrypted import EncryptedVaultBackend
from homie_core.backend.state import StateBackend
from homie_core.backend.protocol import EditResult, FileContent

# Fixed 32-byte key for deterministic tests
_FIXED_KEY = b"\x00" * 32


def _key_provider() -> bytes:
    return _FIXED_KEY


def make_encrypted_backend() -> tuple[EncryptedVaultBackend, StateBackend]:
    inner = StateBackend()
    backend = EncryptedVaultBackend(inner=inner, key_provider=_key_provider)
    return backend, inner


# ---------------------------------------------------------------------------
# Write / Read roundtrip
# ---------------------------------------------------------------------------

class TestWriteReadRoundtrip:
    def test_roundtrip_simple(self):
        backend, _ = make_encrypted_backend()
        backend.write("/secret.txt", "hello world")
        fc = backend.read("/secret.txt")
        assert fc.content == "hello world"

    def test_roundtrip_multiline(self):
        backend, _ = make_encrypted_backend()
        content = "line1\nline2\nline3"
        backend.write("/multi.txt", content)
        fc = backend.read("/multi.txt")
        assert fc.content == content

    def test_roundtrip_empty_content(self):
        backend, _ = make_encrypted_backend()
        backend.write("/empty.txt", "")
        fc = backend.read("/empty.txt")
        assert fc.content == ""

    def test_roundtrip_unicode(self):
        backend, _ = make_encrypted_backend()
        content = "привет мир\n日本語\nemoji: 🔐"
        backend.write("/unicode.txt", content)
        fc = backend.read("/unicode.txt")
        assert fc.content == content

    def test_read_applies_offset_and_limit(self):
        backend, _ = make_encrypted_backend()
        backend.write("/paged.txt", "a\nb\nc\nd\ne")
        fc = backend.read("/paged.txt", offset=1, limit=2)
        assert fc.content == "b\nc"
        assert fc.total_lines == 5
        assert fc.truncated is True


# ---------------------------------------------------------------------------
# Inner stores encrypted (not plaintext)
# ---------------------------------------------------------------------------

class TestInnerStoresEncrypted:
    def test_inner_does_not_contain_plaintext(self):
        backend, inner = make_encrypted_backend()
        backend.write("/sensitive.txt", "my secret password")
        # Read raw bytes from inner backend
        raw_fc = inner.read("/sensitive.txt", offset=0, limit=10_000)
        assert "my secret password" not in raw_fc.content

    def test_inner_content_is_base64(self):
        import base64
        backend, inner = make_encrypted_backend()
        backend.write("/data.txt", "test data")
        raw_fc = inner.read("/data.txt", offset=0, limit=10_000)
        # Should be valid base64
        raw_fc.content  # may span multiple lines; join if needed
        try:
            base64.b64decode(raw_fc.content)
        except Exception:
            pytest.fail("Inner content is not valid base64")

    def test_two_writes_produce_different_ciphertext(self):
        """Each write uses a fresh random nonce, so ciphertexts differ."""
        backend, inner = make_encrypted_backend()
        backend.write("/f.txt", "same content")
        ct1 = inner.read("/f.txt", offset=0, limit=10_000).content
        backend.write("/f.txt", "same content")
        ct2 = inner.read("/f.txt", offset=0, limit=10_000).content
        # Different nonces → different ciphertexts (with overwhelming probability)
        assert ct1 != ct2


# ---------------------------------------------------------------------------
# Edit through encryption
# ---------------------------------------------------------------------------

class TestEdit:
    def test_edit_success(self):
        backend, _ = make_encrypted_backend()
        backend.write("/code.txt", "x = 1\ny = 2")
        result = backend.edit("/code.txt", "x = 1", "x = 99")
        assert result.success is True
        fc = backend.read("/code.txt")
        assert "x = 99" in fc.content
        assert "x = 1" not in fc.content

    def test_edit_replace_all(self):
        backend, _ = make_encrypted_backend()
        backend.write("/dup.txt", "foo\nfoo\nfoo")
        result = backend.edit("/dup.txt", "foo", "bar", replace_all=True)
        assert result.success is True
        fc = backend.read("/dup.txt")
        assert "foo" not in fc.content
        assert fc.content == "bar\nbar\nbar"

    def test_edit_not_unique_fails(self):
        backend, _ = make_encrypted_backend()
        backend.write("/dup.txt", "foo\nfoo")
        result = backend.edit("/dup.txt", "foo", "bar")
        assert result.success is False
        assert result.occurrences == 2

    def test_edit_string_not_found(self):
        backend, _ = make_encrypted_backend()
        backend.write("/f.txt", "hello")
        result = backend.edit("/f.txt", "nothere", "x")
        assert result.success is False
        assert result.occurrences == 0

    def test_edit_file_not_found(self):
        backend, _ = make_encrypted_backend()
        result = backend.edit("/missing.txt", "x", "y")
        assert result.success is False
        assert result.error is not None

    def test_edit_ciphertext_updated_in_inner(self):
        backend, inner = make_encrypted_backend()
        backend.write("/f.txt", "hello world")
        ct_before = inner.read("/f.txt", offset=0, limit=10_000).content
        backend.edit("/f.txt", "hello", "goodbye")
        ct_after = inner.read("/f.txt", offset=0, limit=10_000).content
        assert ct_before != ct_after
        # Verify decrypted correctly
        fc = backend.read("/f.txt")
        assert fc.content == "goodbye world"


# ---------------------------------------------------------------------------
# ls and glob delegate to inner
# ---------------------------------------------------------------------------

class TestLsGlob:
    def test_ls_delegates(self):
        backend, _ = make_encrypted_backend()
        backend.write("/dir/file.txt", "content")
        entries = backend.ls("/dir")
        assert any(e.name == "file.txt" for e in entries)

    def test_glob_delegates(self):
        backend, _ = make_encrypted_backend()
        backend.write("/a.txt", "a")
        backend.write("/b.py", "b")
        matches = backend.glob("*.txt")
        assert any("a.txt" in m for m in matches)
        assert not any("b.py" in m for m in matches)


# ---------------------------------------------------------------------------
# grep always returns empty
# ---------------------------------------------------------------------------

class TestGrep:
    def test_grep_returns_empty(self):
        backend, _ = make_encrypted_backend()
        backend.write("/secret.txt", "password: 12345")
        results = backend.grep("password")
        assert results == []

    def test_grep_returns_empty_even_with_match_in_plaintext(self):
        backend, _ = make_encrypted_backend()
        backend.write("/data.txt", "findme")
        results = backend.grep("findme")
        assert results == []
