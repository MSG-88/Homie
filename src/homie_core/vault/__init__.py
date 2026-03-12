"""Homie Secure Vault — encrypted local storage for credentials and user data."""

from homie_core.vault.exceptions import (
    VaultError,
    VaultLockedError,
    VaultAuthError,
    VaultCorruptError,
    CredentialNotFoundError,
    RateLimitError,
)
from homie_core.vault.secure_vault import SecureVault

__all__ = [
    "SecureVault",
    "VaultError",
    "VaultLockedError",
    "VaultAuthError",
    "VaultCorruptError",
    "CredentialNotFoundError",
    "RateLimitError",
]
