"""One-time provisioning of the internal cloud AI key.

Run once to store the Qubrid API key in the vault:
    python -m homie_core.cloud_ai_provision
"""
from __future__ import annotations

from homie_core.vault.secure_vault import SecureVault
from homie_core.cloud_ai import store_cloud_key, is_configured


_QUBRID_KEY = "k_49ec3b193777.ag6sAWo4URpM8_KV1GUJh7PRwQR-7-VvZYTKU6u05RIwKWKk6EvWzw"


def provision(vault: SecureVault | None = None) -> bool:
    """Store the internal cloud AI key in the vault. Returns True if newly stored."""
    own_vault = vault is None
    if own_vault:
        vault = SecureVault()
        vault.unlock()
    try:
        if is_configured(vault):
            return False
        store_cloud_key(vault, _QUBRID_KEY)
        return True
    finally:
        if own_vault:
            vault.lock()


if __name__ == "__main__":
    stored = provision()
    if stored:
        print("Internal cloud AI key provisioned successfully.")
    else:
        print("Internal cloud AI key already configured.")
