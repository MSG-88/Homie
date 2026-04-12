from homie_core.mesh.node_context import NodeContext
from homie_core.mesh.unified_user_model import UnifiedUserModel


def test_empty_model():
    m = UnifiedUserModel()
    assert m.active_nodes == [] and m.primary_node is None


def test_update_single_node():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=30.0, flow_score=0.8))
    assert m.active_nodes == ["desktop"] and m.primary_node == "desktop"


def test_update_multiple_nodes():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=30.0))
    m.update_node(NodeContext(node_id="laptop", node_name="laptop", activity_type="browsing", minutes_active=5.0))
    assert len(m.active_nodes) == 2 and m.primary_node == "desktop"


def test_idle_node_not_primary():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="desktop", node_name="desktop", idle_minutes=10.0, minutes_active=0.0))
    m.update_node(NodeContext(node_id="laptop", node_name="laptop", activity_type="coding", minutes_active=5.0))
    assert m.primary_node == "laptop"


def test_activity_summary():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", active_window="VS Code"))
    assert "coding" in m.activity_summary() and "desktop" in m.activity_summary()


def test_to_context_block():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=30.0, flow_score=0.85))
    m.update_node(NodeContext(node_id="laptop", node_name="laptop", idle_minutes=10.0, minutes_active=0.0))
    block = m.to_context_block()
    assert "[CROSS-DEVICE CONTEXT]" in block and "desktop" in block


def test_node_contexts_dict():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="n1", node_name="box1"))
    m.update_node(NodeContext(node_id="n2", node_name="box2"))
    assert len(m.node_contexts) == 2


def test_remove_stale_node():
    m = UnifiedUserModel()
    m.update_node(NodeContext(node_id="n1", node_name="box"))
    m.remove_node("n1")
    assert len(m.node_contexts) == 0
