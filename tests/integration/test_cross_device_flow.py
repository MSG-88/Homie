"""Integration tests: Cross-Device Context Flow."""
import time
from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.node_context import NodeContext
from homie_core.mesh.unified_user_model import UnifiedUserModel
from homie_core.mesh.context_handoff import ContextHandoff
from homie_core.mesh.cross_device_perceiver import CrossDevicePerceiver


def test_full_cross_device_context_flow(tmp_path):
    desktop_id = NodeIdentity.generate()
    laptop_id = NodeIdentity.generate()
    desktop_mgr = MeshManager(identity=desktop_id, data_dir=tmp_path / "desktop")
    laptop_mgr = MeshManager(identity=laptop_id, data_dir=tmp_path / "laptop")
    desktop_ctx = NodeContext(node_id=desktop_id.node_id, node_name="desktop",
                              activity_type="coding", active_window="VS Code — mesh.py",
                              minutes_active=45.0, flow_score=0.85)
    laptop_ctx = NodeContext(node_id=laptop_id.node_id, node_name="laptop",
                             activity_type="browsing", active_window="Chrome — docs", minutes_active=10.0)
    desktop_mgr.emit("context", "activity_changed", desktop_ctx.to_event_payload())
    laptop_mgr.emit("context", "activity_changed", laptop_ctx.to_event_payload())
    model = UnifiedUserModel()
    model.update_node(desktop_ctx)
    model.update_node(laptop_ctx)
    assert len(model.active_nodes) == 2
    assert model.primary_node == desktop_id.node_id
    perceiver = CrossDevicePerceiver(unified_model=model)
    block = perceiver.get_context_block()
    assert "CROSS-DEVICE" in block and "desktop" in block and "laptop" in block and "primary" in block


def test_device_switch_handoff_flow(tmp_path):
    handoff = ContextHandoff()
    model = UnifiedUserModel()
    perceiver = CrossDevicePerceiver(unified_model=model)
    laptop_active = NodeContext(node_id="laptop", node_name="laptop", activity_type="coding",
                                active_window="VS Code — api.py", minutes_active=30.0)
    desktop_idle = NodeContext(node_id="desktop", node_name="desktop", idle_minutes=10.0, minutes_active=0.0)
    model.update_node(laptop_active); model.update_node(desktop_idle)
    result = handoff.check("desktop", desktop_idle, {"laptop": laptop_active})
    assert result is None
    laptop_idle = NodeContext(node_id="laptop", node_name="laptop", idle_minutes=2.0, minutes_active=0.0)
    desktop_active = NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=1.0)
    model.update_node(laptop_idle); model.update_node(desktop_active)
    result = handoff.check("desktop", desktop_active, {"laptop": laptop_idle})
    assert result is not None and result["from_node"] == "laptop" and "coding" in result["previous_activity"]
    perceiver.set_pending_handoff(result)
    block = perceiver.get_context_block()
    assert "DEVICE SWITCH" in block and "laptop" in block


def test_context_events_sync_between_nodes(tmp_path):
    from homie_core.mesh.sync_protocol import SyncRequest
    node_a = NodeIdentity.generate()
    node_b = NodeIdentity.generate()
    mgr_a = MeshManager(identity=node_a, data_dir=tmp_path / "a")
    mgr_b = MeshManager(identity=node_b, data_dir=tmp_path / "b")
    ctx = NodeContext(node_id=node_a.node_id, node_name="node-a", activity_type="coding", minutes_active=20.0)
    mgr_a.emit("context", "activity_changed", ctx.to_event_payload())
    req = SyncRequest(node_id=node_b.node_id, last_event_id=None, vector_clock={})
    resp = mgr_a.handle_sync_request(req)
    assert mgr_b.apply_sync_response(resp) == 1
    events = mgr_b.events_since(None)
    assert len(events) == 1 and events[0].category == "context"
    restored = NodeContext.from_dict(events[0].payload)
    assert restored.activity_type == "coding" and restored.node_name == "node-a"
