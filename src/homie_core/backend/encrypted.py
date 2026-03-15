"""AES-256-GCM decorator backend — wraps any BackendProtocol with encryption.

Each file's content is stored as base64(nonce_12B || ciphertext || tag_16B).
The ``ls`` and ``glob`` methods are delegated unchanged; ``grep`` always returns
an empty list because encrypted content cannot be searched.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Callable, Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRYPTOGRAPHY_AVAILABLE = False

from homie_core.backend.protocol import (
    BackendProtocol,
    EditResult,
    FileContent,
    FileInfo,
    GrepMatch,
)

logger = logging.getLogger(__name__)

_NONCE_LENGTH = 12


class EncryptedVaultBackend:
    """Decorator that wraps any :class:`BackendProtocol` with AES-256-GCM encryption.

    Parameters
    ----------
    inner:
        The underlying backend where (encrypted) data is stored.
    key_provider:
        Zero-argument callable that returns a 32-byte key each time it is
        called.  The key is fetched fresh for every encrypt/decrypt operation
        so callers can rotate keys or source them from a secrets manager.
    """

    def __init__(
        self,
        inner: BackendProtocol,
        key_provider: Callable[[], bytes],
    ) -> None:
        if not _CRYPTOGRAPHY_AVAILABLE:
            raise ImportError(
                "The 'cryptography' package is required for EncryptedVaultBackend. "
                "Install it with: pip install cryptography"
            )
        self._inner = inner
        self._key_provider = key_provider

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt *plaintext* with AES-256-GCM.

        Returns ``base64(nonce_12B || ciphertext || tag_16B)``.
        """
        key = self._key_provider()
        nonce = os.urandom(_NONCE_LENGTH)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ct).decode("ascii")

    def _decrypt(self, data: str) -> str:
        """Decrypt a base64-encoded AES-256-GCM blob produced by :meth:`_encrypt`."""
        key = self._key_provider()
        raw = base64.b64decode(data)
        nonce = raw[:_NONCE_LENGTH]
        ct = raw[_NONCE_LENGTH:]
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ct, None)
        return plaintext.decode("utf-8")

    # ------------------------------------------------------------------
    # BackendProtocol
    # ------------------------------------------------------------------

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        """Decrypt the full file then apply *offset*/*limit* on the plaintext lines."""
        # Read the full encrypted blob (use a large limit to get everything).
        encrypted_fc = self._inner.read(path, offset=0, limit=2 ** 31 - 1)
        if not encrypted_fc.content:
            return FileContent(content="", total_lines=0, truncated=False)
        plaintext = self._decrypt(encrypted_fc.content)
        lines = plaintext.splitlines()
        total = len(lines)
        sliced = lines[offset: offset + limit]
        truncated = (offset + limit) < total
        return FileContent(
            content="\n".join(sliced),
            total_lines=total,
            truncated=truncated,
        )

    def write(self, path: str, content: str) -> None:
        """Encrypt *content* then store it in the inner backend."""
        encrypted = self._encrypt(content)
        self._inner.write(path, encrypted)

    def edit(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Read-decrypt-edit-encrypt-write cycle for in-place editing."""
        # Decrypt current content
        try:
            encrypted_fc = self._inner.read(path, offset=0, limit=2 ** 31 - 1)
        except FileNotFoundError:
            return EditResult(success=False, occurrences=0, error=f"File not found: {path!r}")

        plaintext = self._decrypt(encrypted_fc.content) if encrypted_fc.content else ""

        # Perform the edit in plaintext
        count = plaintext.count(old)
        if count == 0:
            return EditResult(success=False, occurrences=0, error=f"String not found in {path!r}")
        if not replace_all and count > 1:
            return EditResult(
                success=False,
                occurrences=count,
                error=(
                    f"String appears {count} times in {path!r}; "
                    "use replace_all=True to replace all occurrences"
                ),
            )

        new_plaintext = plaintext.replace(old, new)
        self._inner.write(path, self._encrypt(new_plaintext))
        return EditResult(success=True, occurrences=count)

    def ls(self, path: str = "/") -> list[FileInfo]:
        """Delegate to the inner backend (paths are not encrypted)."""
        return self._inner.ls(path)

    def glob(self, pattern: str) -> list[str]:
        """Delegate to the inner backend (paths are not encrypted)."""
        return self._inner.glob(pattern)

    def grep(
        self,
        pattern: str,
        path: str = "/",
        include: Optional[str] = None,
    ) -> list[GrepMatch]:
        """Always returns an empty list — encrypted content cannot be searched."""
        logger.debug(
            "EncryptedVaultBackend.grep called but returning empty list "
            "(content is encrypted; pattern=%r, path=%r)", pattern, path
        )
        return []
