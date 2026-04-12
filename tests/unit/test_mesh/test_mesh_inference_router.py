from unittest.mock import MagicMock
from homie_core.mesh.mesh_inference_router import MeshInferenceRouter

def _mock_engine(loaded=True, resp="local response"):
    e = MagicMock(); e.is_loaded = loaded; e.generate.return_value = resp; e.stream.return_value = iter([resp]); return e

def _mock_client(avail=True, resp="hub response"):
    c = MagicMock(); c.is_available = avail; c.generate.return_value = resp; c.stream.return_value = iter([resp]); return c

def test_local_model_used_first():
    engine, client = _mock_engine(True, "local"), _mock_client(True, "hub")
    r = MeshInferenceRouter(model_engine=engine, mesh_client=client, priority=["local","hub"])
    assert r.generate("p") == "local"
    client.generate.assert_not_called()

def test_fallback_to_hub():
    engine, client = _mock_engine(False), _mock_client(True, "from hub")
    assert MeshInferenceRouter(model_engine=engine, mesh_client=client, priority=["local","hub"]).generate("p") == "from hub"

def test_fallback_to_qubrid():
    engine, client = _mock_engine(False), _mock_client(False)
    qubrid = MagicMock(); qubrid.is_available = True; qubrid.generate.return_value = "cloud"
    assert MeshInferenceRouter(model_engine=engine, mesh_client=client, qubrid_client=qubrid, priority=["local","hub","qubrid"]).generate("p") == "cloud"

def test_all_unavailable_raises():
    try:
        MeshInferenceRouter(model_engine=_mock_engine(False), mesh_client=_mock_client(False), priority=["local","hub"]).generate("p")
        assert False
    except RuntimeError as e: assert "unavailable" in str(e).lower()

def test_stream_uses_local():
    engine, client = _mock_engine(True, "token"), _mock_client(True)
    assert "".join(MeshInferenceRouter(model_engine=engine, mesh_client=client, priority=["local","hub"]).stream("p")) == "token"
    engine.stream.assert_called_once()

def test_stream_fallback_to_hub():
    engine, client = _mock_engine(False), _mock_client(True, "hub token")
    assert "".join(MeshInferenceRouter(model_engine=engine, mesh_client=client, priority=["local","hub"]).stream("p")) == "hub token"

def test_active_source():
    engine, client = _mock_engine(True), _mock_client(True)
    r = MeshInferenceRouter(model_engine=engine, mesh_client=client, priority=["local","hub"])
    assert r.active_source == "Local"
    engine.is_loaded = False
    assert r.active_source == "Mesh Hub"
    client.is_available = False
    assert r.active_source == "None"
