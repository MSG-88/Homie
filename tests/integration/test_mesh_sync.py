"""Integration test: two-node sync via MeshManager."""
import time
from homie_core.mesh.events import HomieEvent
from homie_core.mesh.sync_protocol import SyncRequest, SyncResponse
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.identity import NodeIdentity


def test_mesh_manager_initialization(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    assert mgr.node_id == identity.node_id
    assert mgr.event_count() == 0


def test_mesh_manager_emit_event(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    mgr.emit("memory", "fact_learned", {"fact": "user likes tea"})
    assert mgr.event_count() == 1
    events = mgr.events_since(None)
    assert events[0].category == "memory"
    assert events[0].payload["fact"] == "user likes tea"
    assert events[0].node_id == identity.node_id


def test_two_node_sync_simulation(tmp_path):
    hub_identity = NodeIdentity.generate()
    hub_mgr = MeshManager(identity=hub_identity, data_dir=tmp_path / "hub")
    spoke_identity = NodeIdentity.generate()
    spoke_mgr = MeshManager(identity=spoke_identity, data_dir=tmp_path / "spoke")
    for i in range(3):
        hub_mgr.emit("memory", "fact_learned", {"fact": f"fact-{i}"})
        time.sleep(0.002)
    for i in range(2):
        spoke_mgr.emit("preference", "feedback", {"rating": i})
        time.sleep(0.002)
    # Spoke requests all events from Hub (first sync, never seen Hub events)
    first_sync_request = SyncRequest(node_id=spoke_identity.node_id, last_event_id=None, vector_clock={})
    response = hub_mgr.handle_sync_request(first_sync_request)
    assert len(response.events) == 3
    applied = spoke_mgr.apply_sync_response(response)
    assert applied == 3
    assert spoke_mgr.event_count() == 5  # 2 own + 3 from Hub
    spoke_events = spoke_mgr.get_unsynced_for_hub()
    assert len(spoke_events) == 2


def test_idempotent_sync(tmp_path):
    hub = MeshManager(identity=NodeIdentity.generate(), data_dir=tmp_path / "hub")
    spoke = MeshManager(identity=NodeIdentity.generate(), data_dir=tmp_path / "spoke")
    hub.emit("task", "created", {"task": "test"})
    req = SyncRequest(node_id=spoke.node_id, last_event_id=None, vector_clock={})
    resp = hub.handle_sync_request(req)
    spoke.apply_sync_response(resp)
    spoke.apply_sync_response(resp)  # Again
    assert spoke.event_count() == 1
