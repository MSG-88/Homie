"""Unit tests for ModelDistributor."""
from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.model_distributor import ModelDistributor


def test_announce_model_update(tmp_path):
    mgr = MeshManager(identity=NodeIdentity.generate(), data_dir=tmp_path)
    dist = ModelDistributor(mesh_manager=mgr)
    dist.announce_update(model_name="homie-v2", model_path="/models/v2.gguf", score_improvement=0.05, cycle=3)
    e = mgr.events_since(None)
    assert len(e) == 1 and e[0].category == "learning" and e[0].event_type == "model_updated"
    assert e[0].payload["model_name"] == "homie-v2"


def test_announce_training_started(tmp_path):
    mgr = MeshManager(identity=NodeIdentity.generate(), data_dir=tmp_path)
    dist = ModelDistributor(mesh_manager=mgr)
    dist.announce_training_started(cycle=4, sft_pairs=500, dpo_pairs=100)
    assert mgr.events_since(None)[0].event_type == "training_started"


def test_announce_training_completed(tmp_path):
    mgr = MeshManager(identity=NodeIdentity.generate(), data_dir=tmp_path)
    dist = ModelDistributor(mesh_manager=mgr)
    dist.announce_training_completed(cycle=4, score=0.82, promoted=True)
    assert mgr.events_since(None)[0].payload["promoted"] is True


def test_get_model_history(tmp_path):
    mgr = MeshManager(identity=NodeIdentity.generate(), data_dir=tmp_path)
    dist = ModelDistributor(mesh_manager=mgr)
    dist.announce_update(model_name="v1", model_path="/v1", score_improvement=0.03, cycle=1)
    dist.announce_update(model_name="v2", model_path="/v2", score_improvement=0.05, cycle=2)
    assert len(dist.get_model_history()) == 2
