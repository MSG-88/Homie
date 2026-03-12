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

_KEY_LENGTH = 32
_NONCE_LENGTH = 12


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
    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt_field(ciphertext_b64: str, key: bytearray) -> str:
    """Decrypt a base64-encoded AES-256-GCM field.

    Raises VaultCorruptError if decryption fails (wrong key or tampered data).
    """
    try:
        raw = base64.b64decode(ciphertext_b64)
    except Exception as e:
        raise VaultCorruptError(f"Invalid base64 ciphertext: {e}") from e

    if len(raw) < _NONCE_LENGTH + 16:
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
