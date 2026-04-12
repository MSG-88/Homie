import time
from homie_core.mesh.pairing import (
    PairingSession, generate_pairing_code, verify_pairing_code, derive_shared_secret,
)

def test_generate_pairing_code():
    session = generate_pairing_code(ttl_seconds=300)
    assert len(session.code) == 6
    assert session.code.isdigit()
    assert session.expires_at > time.time()
    assert session.hub_public_key_pem != ""

def test_verify_valid_code():
    session = generate_pairing_code(ttl_seconds=300)
    assert verify_pairing_code(session, session.code) is True

def test_verify_wrong_code():
    session = generate_pairing_code(ttl_seconds=300)
    assert verify_pairing_code(session, "000000") is False

def test_verify_expired_code():
    session = generate_pairing_code(ttl_seconds=0)
    assert verify_pairing_code(session, session.code) is False

def test_derive_shared_secret():
    session_a = generate_pairing_code(ttl_seconds=300)
    session_b = generate_pairing_code(ttl_seconds=300)
    secret_a = derive_shared_secret(
        session_a.hub_private_key_x25519_bytes,
        session_b.spoke_x25519_public_bytes,
    )
    secret_b = derive_shared_secret(
        session_b.hub_private_key_x25519_bytes,
        session_a.spoke_x25519_public_bytes,
    )
    assert len(secret_a) == 32
    assert len(secret_b) == 32
