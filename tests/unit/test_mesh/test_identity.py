import json
from pathlib import Path
from homie_core.mesh.identity import NodeIdentity, load_identity, save_identity


def test_node_identity_creation():
    identity = NodeIdentity.generate()
    assert len(identity.node_id) == 36
    assert identity.node_name != ""
    assert identity.role == "standalone"
    assert identity.mesh_id is None
    assert identity.created_at != ""


def test_node_identity_deterministic():
    a = NodeIdentity.generate()
    b = NodeIdentity.generate()
    assert a.node_id != b.node_id


def test_save_and_load_identity(tmp_path):
    path = tmp_path / "node.json"
    identity = NodeIdentity.generate()
    identity.node_name = "test-box"
    save_identity(identity, path)
    loaded = load_identity(path)
    assert loaded.node_id == identity.node_id
    assert loaded.node_name == "test-box"
    assert loaded.role == "standalone"
    assert loaded.created_at == identity.created_at


def test_load_identity_missing_file(tmp_path):
    path = tmp_path / "nope.json"
    assert load_identity(path) is None


def test_identity_has_ed25519_keypair():
    identity = NodeIdentity.generate()
    assert identity.public_key_pem != ""
    assert identity.private_key_pem != ""
    assert "BEGIN PUBLIC KEY" in identity.public_key_pem
    assert "BEGIN PRIVATE KEY" in identity.private_key_pem


def test_saved_identity_excludes_private_key(tmp_path):
    path = tmp_path / "node.json"
    identity = NodeIdentity.generate()
    save_identity(identity, path)
    raw = json.loads(path.read_text())
    assert "private_key_pem" not in raw
    assert "public_key_pem" in raw
