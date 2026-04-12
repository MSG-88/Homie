"""Smoke test: verify mesh foundation components work end-to-end."""
from pathlib import Path

from homie_core.mesh.identity import NodeIdentity, save_identity, load_identity
from homie_core.mesh.capabilities import detect_capabilities
from homie_core.mesh.election import ElectionCandidate, elect_hub
from homie_core.mesh.pairing import generate_pairing_code, verify_pairing_code
from homie_core.mesh.registry import MeshNodeRegistry, MeshNodeRecord
from homie_core.platform.detect import get_platform_adapter
from homie_core.config import HomieConfig


def test_full_mesh_foundation_flow(tmp_path):
    """End-to-end: identity -> capabilities -> election -> registry -> config."""
    # 1. Generate identity
    identity = NodeIdentity.generate()
    assert identity.node_id

    # 2. Save and reload
    node_path = tmp_path / "node.json"
    save_identity(identity, node_path)
    loaded = load_identity(node_path)
    assert loaded.node_id == identity.node_id

    # 3. Detect capabilities
    caps = detect_capabilities()
    assert caps.cpu_cores > 0

    # 4. Election with this node as candidate
    candidate = ElectionCandidate(
        node_id=identity.node_id,
        capability_score=caps.capability_score(),
        created_at=identity.created_at,
    )
    winner = elect_hub([candidate])
    assert winner.node_id == identity.node_id

    # 5. Registry
    db_path = tmp_path / "mesh.db"
    registry = MeshNodeRegistry(db_path)
    registry.initialize()
    registry.upsert(MeshNodeRecord(
        node_id=identity.node_id,
        node_name=identity.node_name,
        role="hub",
        mesh_id="test-mesh",
        capability_score=caps.capability_score(),
        capabilities_json="{}",
        lan_ip="192.168.1.1",
        tailnet_ip="",
        public_key_ed25519=identity.public_key_pem,
        status="online",
    ))
    node = registry.get(identity.node_id)
    assert node.role == "hub"

    # 6. Pairing
    session = generate_pairing_code(ttl_seconds=60)
    assert verify_pairing_code(session, session.code) is True

    # 7. Platform adapter
    adapter = get_platform_adapter()
    assert adapter.get_hostname() != ""

    # 8. Config has mesh section
    cfg = HomieConfig()
    assert cfg.mesh.enabled is True


def test_two_node_election(tmp_path):
    """Simulate two nodes: desktop (GPU) should win hub election over laptop."""
    desktop = ElectionCandidate(
        node_id="desktop-1",
        capability_score=290.0,
        created_at="2026-01-01T00:00:00",
    )
    laptop = ElectionCandidate(
        node_id="laptop-1",
        capability_score=40.0,
        created_at="2026-01-02T00:00:00",
    )
    winner = elect_hub([desktop, laptop])
    assert winner.node_id == "desktop-1"
