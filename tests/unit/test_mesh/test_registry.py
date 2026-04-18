from homie_core.mesh.registry import MeshNodeRegistry, MeshNodeRecord


def test_registry_initialize(tmp_path):
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()


def test_register_and_get_node(tmp_path):
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()
    record = MeshNodeRecord(
        node_id="abc-123", node_name="test-box", role="spoke",
        mesh_id="mesh-1", capability_score=42.0, capabilities_json='{"cpu_cores": 4}',
        lan_ip="192.168.1.10", tailnet_ip="", public_key_ed25519="PEM...", status="online",
    )
    reg.upsert(record)
    loaded = reg.get("abc-123")
    assert loaded is not None
    assert loaded.node_name == "test-box"
    assert loaded.capability_score == 42.0
    assert loaded.status == "online"


def test_upsert_updates_existing(tmp_path):
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()
    record = MeshNodeRecord(
        node_id="abc-123", node_name="box-v1", role="spoke",
        mesh_id="m1", capability_score=10.0, capabilities_json="{}",
        lan_ip="", tailnet_ip="", public_key_ed25519="", status="online",
    )
    reg.upsert(record)
    record.node_name = "box-v2"
    record.capability_score = 99.0
    reg.upsert(record)
    all_nodes = reg.list_all()
    assert len(all_nodes) == 1
    assert all_nodes[0].node_name == "box-v2"
    assert all_nodes[0].capability_score == 99.0


def test_list_all_nodes(tmp_path):
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()
    for i in range(3):
        reg.upsert(MeshNodeRecord(
            node_id=f"node-{i}", node_name=f"box-{i}", role="spoke",
            mesh_id="m1", capability_score=float(i), capabilities_json="{}",
            lan_ip="", tailnet_ip="", public_key_ed25519="", status="online",
        ))
    nodes = reg.list_all()
    assert len(nodes) == 3


def test_remove_node(tmp_path):
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()
    reg.upsert(MeshNodeRecord(
        node_id="gone", node_name="bye", role="spoke",
        mesh_id="m1", capability_score=0, capabilities_json="{}",
        lan_ip="", tailnet_ip="", public_key_ed25519="", status="online",
    ))
    assert reg.get("gone") is not None
    reg.remove("gone")
    assert reg.get("gone") is None


def test_update_status(tmp_path):
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()
    reg.upsert(MeshNodeRecord(
        node_id="n1", node_name="box", role="spoke",
        mesh_id="m1", capability_score=10, capabilities_json="{}",
        lan_ip="", tailnet_ip="", public_key_ed25519="", status="online",
    ))
    reg.update_status("n1", "offline")
    node = reg.get("n1")
    assert node.status == "offline"
