from homie_core.config import HomieConfig
from homie_core.mesh.bootstrap import bootstrap_mesh
from homie_core.mesh.health import MeshHealthChecker, SystemHealth, HealthCheck


def test_health_check_all_pass(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    checker = MeshHealthChecker(mesh_context=ctx)
    health = checker.run_all()
    assert health.healthy is True
    assert health.status == "healthy"
    assert len(health.checks) == 6


def test_health_check_no_context():
    checker = MeshHealthChecker(mesh_context=None)
    health = checker.run_all()
    assert health.healthy is False
    assert "degraded" in health.status


def test_health_summary(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    checker = MeshHealthChecker(mesh_context=ctx)
    health = checker.run_all()
    summary = health.summary()
    assert "healthy" in summary
    assert "identity" in summary
    assert "event_store" in summary


def test_health_check_has_latency(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    checker = MeshHealthChecker(mesh_context=ctx)
    health = checker.run_all()
    for check in health.checks:
        assert check.latency_ms >= 0


def test_individual_checks(tmp_path):
    cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
    ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
    checker = MeshHealthChecker(mesh_context=ctx)
    identity = checker._check_identity()
    assert identity.healthy and "node=" in identity.message
    events = checker._check_event_store()
    assert events.healthy
    caps = checker._check_capabilities()
    assert caps.healthy and "score=" in caps.message
