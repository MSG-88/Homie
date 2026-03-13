"""Internal cloud AI configuration — transparent fallback to Qubrid platform."""
from __future__ import annotations

from typing import Optional

from homie_core.vault.secure_vault import SecureVault

# Internal defaults — users don't configure these directly.
_PROVIDER = "qubrid"
_ACCOUNT_ID = "internal"
_BASE_URL = "https://platform.qubrid.com/v1"
_DEFAULT_MODEL = "Qwen/Qwen3.5-Flash"


def store_cloud_key(vault: SecureVault, api_key: str) -> str:
    """Store the internal cloud AI API key in the vault."""
    return vault.store_credential(
        provider=_PROVIDER,
        account_id=_ACCOUNT_ID,
        token_type="api_key",
        access_token=api_key,
    )


def get_cloud_config(vault: SecureVault) -> Optional[dict]:
    """Return cloud AI config dict if an internal key is stored, else None.

    Returns ``{"api_key": ..., "base_url": ..., "model": ...}`` or *None*.
    """
    try:
        cred = vault.get_credential(_PROVIDER, _ACCOUNT_ID)
        if cred and cred.active:
            return {
                "api_key": cred.access_token,
                "base_url": _BASE_URL,
                "model": _DEFAULT_MODEL,
            }
    except Exception:
        pass
    return None


def is_configured(vault: SecureVault) -> bool:
    """Check whether the internal cloud AI key is stored."""
    return get_cloud_config(vault) is not None
