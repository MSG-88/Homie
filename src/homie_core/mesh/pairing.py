"""Pairing protocol — one-time code + X25519 key exchange for mesh auth."""
from __future__ import annotations
import secrets, time
from dataclasses import dataclass
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives import serialization

@dataclass
class PairingSession:
    code: str
    expires_at: float
    hub_public_key_pem: str
    hub_private_key_x25519_bytes: bytes
    spoke_x25519_public_bytes: bytes
    used: bool = False

def generate_pairing_code(ttl_seconds: int = 300) -> PairingSession:
    code = f"{secrets.randbelow(1000000):06d}"
    ed_private = Ed25519PrivateKey.generate()
    ed_public_pem = ed_private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    x_private = X25519PrivateKey.generate()
    x_public_bytes = x_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw,
    )
    x_private_bytes = x_private.private_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return PairingSession(
        code=code, expires_at=time.time() + ttl_seconds,
        hub_public_key_pem=ed_public_pem,
        hub_private_key_x25519_bytes=x_private_bytes,
        spoke_x25519_public_bytes=x_public_bytes,
        used=False,
    )

def verify_pairing_code(session: PairingSession, code: str) -> bool:
    if session.used:
        return False
    if time.time() >= session.expires_at:
        return False
    return secrets.compare_digest(session.code, code)

def derive_shared_secret(my_x25519_private_bytes: bytes, peer_x25519_public_bytes: bytes) -> bytes:
    my_private = X25519PrivateKey.from_private_bytes(my_x25519_private_bytes)
    peer_public = X25519PublicKey.from_public_bytes(peer_x25519_public_bytes)
    return my_private.exchange(peer_public)
