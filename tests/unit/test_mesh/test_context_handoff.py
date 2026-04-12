from homie_core.mesh.node_context import NodeContext
from homie_core.mesh.context_handoff import ContextHandoff


def test_no_handoff_on_first_activity():
    handoff = ContextHandoff()
    result = handoff.check(
        "desktop",
        NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=5.0),
        {},
    )
    assert result is None


def test_detects_switch_from_peer():
    handoff = ContextHandoff()
    laptop_active = NodeContext(
        node_id="laptop",
        node_name="laptop",
        activity_type="coding",
        active_window="VS Code — sync.py",
        minutes_active=30.0,
        idle_minutes=0.0,
    )
    handoff.check(
        "desktop",
        NodeContext(node_id="desktop", node_name="desktop", idle_minutes=10.0, minutes_active=0.0),
        {"laptop": laptop_active},
    )
    laptop_idle = NodeContext(node_id="laptop", node_name="laptop", idle_minutes=2.0, minutes_active=0.0)
    result = handoff.check(
        "desktop",
        NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=1.0),
        {"laptop": laptop_idle},
    )
    assert result is not None
    assert result["from_node"] == "laptop"
    assert "coding" in result["previous_activity"]


def test_no_duplicate_handoff():
    handoff = ContextHandoff()
    handoff.check(
        "desktop",
        NodeContext(node_id="desktop", node_name="desktop", idle_minutes=5.0, minutes_active=0.0),
        {
            "laptop": NodeContext(
                node_id="laptop", node_name="laptop", activity_type="coding", minutes_active=20.0
            )
        },
    )
    desktop_active = NodeContext(
        node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=1.0
    )
    laptop_idle = NodeContext(node_id="laptop", node_name="laptop", idle_minutes=2.0, minutes_active=0.0)
    r1 = handoff.check("desktop", desktop_active, {"laptop": laptop_idle})
    assert r1 is not None
    r2 = handoff.check("desktop", desktop_active, {"laptop": laptop_idle})
    assert r2 is None


def test_handoff_context_message():
    handoff = ContextHandoff()
    handoff.check(
        "desktop",
        NodeContext(node_id="desktop", node_name="desktop", idle_minutes=5.0, minutes_active=0.0),
        {
            "laptop": NodeContext(
                node_id="laptop",
                node_name="laptop",
                activity_type="browsing",
                active_window="Chrome — Stack Overflow",
                minutes_active=15.0,
            )
        },
    )
    result = handoff.check(
        "desktop",
        NodeContext(node_id="desktop", node_name="desktop", activity_type="coding", minutes_active=1.0),
        {
            "laptop": NodeContext(
                node_id="laptop", node_name="laptop", idle_minutes=2.0, minutes_active=0.0
            )
        },
    )
    assert result is not None
    assert "message" in result
    assert "laptop" in result["message"]
