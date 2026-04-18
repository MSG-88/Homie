from homie_core.mesh.node_context import NodeContext, collect_local_context


def test_node_context_creation():
    ctx = NodeContext(node_id="n1", node_name="desktop", active_window="VS Code — main.py",
                      activity_type="coding", activity_confidence=0.9, minutes_active=45.0, flow_score=0.8,
                      cpu_usage=34.0, ram_usage_gb=18.0, gpu_usage=67.0)
    assert ctx.node_id == "n1" and ctx.activity_type == "coding" and ctx.flow_score == 0.8


def test_node_context_to_dict():
    ctx = NodeContext(node_id="n1", node_name="laptop")
    d = ctx.to_dict()
    assert d["node_id"] == "n1" and "last_updated" in d


def test_node_context_from_dict():
    ctx = NodeContext(node_id="n1", node_name="box", activity_type="browsing")
    assert NodeContext.from_dict(ctx.to_dict()).activity_type == "browsing"


def test_node_context_to_event_payload():
    ctx = NodeContext(node_id="n1", node_name="desktop", active_window="Chrome", activity_type="browsing")
    p = ctx.to_event_payload()
    assert p["activity_type"] == "browsing"


def test_node_context_summary():
    ctx = NodeContext(node_id="n1", node_name="desktop", active_window="VS Code — sync.py",
                      activity_type="coding", minutes_active=30.0, flow_score=0.85)
    s = ctx.summary()
    assert "coding" in s and "desktop" in s


def test_collect_local_context():
    ctx = collect_local_context(node_id="test", node_name="test-box")
    assert ctx.node_id == "test" and ctx.cpu_usage >= 0 and ctx.ram_usage_gb > 0


def test_context_is_idle():
    assert NodeContext(node_id="n1", node_name="n", idle_minutes=10.0).is_idle is True
    assert NodeContext(node_id="n1", node_name="n", idle_minutes=0.0, minutes_active=5.0).is_idle is False
