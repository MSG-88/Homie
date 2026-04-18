from pathlib import Path
from homie_core.config import HomieConfig
from homie_core.mesh.bootstrap import bootstrap_mesh, MeshContext


def test_bootstrap_creates_identity(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    assert ctx.node_id != ""
    assert ctx.node_name != ""
    assert ctx.enabled is True


def test_bootstrap_returns_all_components(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    assert ctx.mesh_manager is not None
    assert ctx.registry is not None
    assert ctx.user_model is not None
    assert ctx.perceiver is not None
    assert ctx.feedback_collector is not None
    assert ctx.task_executor is not None
    assert ctx.auth_store is not None


def test_bootstrap_creates_default_admin(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    users = ctx.auth_store.list_users()
    assert len(users) == 1
    assert users[0].role.name == "ADMIN"


def test_bootstrap_emits_startup_event(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    events = ctx.mesh_manager.events_since(None)
    assert len(events) >= 1
    assert events[0].event_type == "node_started"


def test_bootstrap_idempotent(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx1 = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    ctx2 = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    assert ctx1.node_id == ctx2.node_id  # Same identity reloaded


def test_collect_context(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    node_ctx = ctx.collect_context()
    assert node_ctx.cpu_usage >= 0
    block = ctx.get_cross_device_block()
    assert "CROSS-DEVICE" in block
