from homie_core.mesh.node_context import NodeContext
from homie_core.mesh.unified_user_model import UnifiedUserModel
from homie_core.mesh.cross_device_perceiver import CrossDevicePerceiver


def test_perceiver_with_no_mesh():
    assert CrossDevicePerceiver(unified_model=None).get_context_block() == ""


def test_perceiver_with_single_node():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=30.0))
    block = CrossDevicePerceiver(unified_model=m).get_context_block()
    assert "CROSS-DEVICE" in block and "desktop" in block


def test_perceiver_with_multiple_nodes():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=30.0, flow_score=0.9))
    m.update_node(NodeContext(node_id="laptop", node_name="laptop", idle_minutes=10.0, minutes_active=0.0))
    block = CrossDevicePerceiver(unified_model=m).get_context_block()
    assert "primary" in block and "desktop" in block and "laptop" in block


def test_perceiver_handoff_message():
    p = CrossDevicePerceiver(unified_model=UnifiedUserModel())
    p.set_pending_handoff({"from_node": "laptop", "to_node": "desktop",
                           "message": "You were coding on laptop. Want to continue here?"})
    assert "coding on laptop" in p.get_context_block()


def test_perceiver_clears_handoff_after_use():
    p = CrossDevicePerceiver(unified_model=UnifiedUserModel())
    p.set_pending_handoff({"message": "test handoff"})
    p.get_context_block()
    assert "test handoff" not in p.get_context_block()
