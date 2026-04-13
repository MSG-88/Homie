"""Test mesh API router creation and endpoint structure."""
import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from homie_core.config import HomieConfig
from homie_core.mesh.bootstrap import bootstrap_mesh


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestMeshAPI:
    @pytest.fixture
    def client(self, tmp_path):
        from homie_core.mesh.api import create_mesh_router
        cfg = HomieConfig(storage={"path": str(tmp_path / ".homie")})
        ctx = bootstrap_mesh(cfg, data_dir=tmp_path / "mesh")
        router = create_mesh_router(ctx)
        app = FastAPI()
        app.include_router(router)
        return TestClient(app), ctx

    def test_health(self, client):
        c, _ = client
        resp = c.get("/api/mesh/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_node_status(self, client):
        c, _ = client
        resp = c.get("/api/mesh/node")
        assert resp.status_code == 200
        data = resp.json()
        assert "node_id" in data and "capability_score" in data

    def test_mesh_status(self, client):
        c, _ = client
        resp = c.get("/api/mesh/status")
        assert resp.status_code == 200
        assert "event_count" in resp.json()

    def test_context(self, client):
        c, _ = client
        resp = c.get("/api/mesh/context")
        assert resp.status_code == 200
        assert "context_block" in resp.json()

    def test_events(self, client):
        c, _ = client
        resp = c.get("/api/mesh/events")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_training_status(self, client):
        c, _ = client
        resp = c.get("/api/mesh/training")
        assert resp.status_code == 200
        assert "ready" in resp.json()

    def test_task_requires_auth(self, client):
        c, _ = client
        resp = c.post("/api/mesh/task", json={"target_node": "x", "command": "echo hi"})
        assert resp.status_code == 401

    def test_task_with_auth(self, client):
        c, ctx = client
        users = ctx.auth_store.list_users()
        # Get the admin's API key by creating a new one we know
        _, key = ctx.auth_store.create_user("testuser", role=__import__('homie_core.mesh.auth', fromlist=['Role']).Role.ADMIN)
        resp = c.post("/api/mesh/task",
                       json={"target_node": ctx.identity.node_id, "command": "echo api_test"},
                       headers={"Authorization": f"Bearer {key}"})
        assert resp.status_code == 200
        assert resp.json()["state"] == "completed"
