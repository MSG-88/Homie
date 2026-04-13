from pathlib import Path

from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.feedback_collector import FeedbackCollector, SignalType
from homie_core.mesh.feedback_store import FeedbackStore
from homie_core.mesh.training_trigger import TrainingTrigger
from homie_core.mesh.model_distributor import ModelDistributor
from homie_core.mesh.auto_trainer import AutoTrainer, TrainingCycleResult


def _setup(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path / "mesh")
    fb_store = FeedbackStore(tmp_path / "feedback.db")
    fb_store.initialize()
    trigger = TrainingTrigger(feedback_store=fb_store, min_signals=5)
    dist = ModelDistributor(mesh_manager=mgr)
    trainer = AutoTrainer(
        feedback_store=fb_store, training_trigger=trigger,
        model_distributor=dist, mesh_manager=mgr,
        base_dir=tmp_path / "training",
        modelfile_path=Path(__file__).parent.parent.parent.parent / "Modelfile",
    )
    return mgr, fb_store, trigger, dist, trainer


def test_check_ready_not_enough(tmp_path):
    _, fb_store, _, _, trainer = _setup(tmp_path)
    summary = trainer.check_ready()
    assert summary["ready"] is False
    assert summary["total_signals"] == 0


def test_check_ready_enough(tmp_path):
    _, fb_store, _, _, trainer = _setup(tmp_path)
    collector = FeedbackCollector(node_id="test")
    for i in range(5):
        fb_store.save(collector.record_accepted(query=f"q{i}", response=f"r{i}"))
    summary = trainer.check_ready()
    assert summary["ready"] is True
    assert summary["total_signals"] == 5


def test_export_training_data(tmp_path):
    _, fb_store, _, _, trainer = _setup(tmp_path)
    collector = FeedbackCollector(node_id="test")
    for i in range(3):
        fb_store.save(collector.record_accepted(query=f"q{i}", response=f"good {i}"))
    for i in range(2):
        fb_store.save(collector.record_corrected(query=f"fix{i}", original=f"bad{i}", correction=f"ok{i}"))

    data = trainer.export_training_data()
    assert data["sft_count"] == 3
    assert data["dpo_count"] == 2
    assert Path(data["sft_path"]).exists()
    assert Path(data["dpo_path"]).exists()

    # Verify JSONL content
    import json
    sft_lines = Path(data["sft_path"]).read_text().strip().split("\n")
    assert len(sft_lines) == 3
    first = json.loads(sft_lines[0])
    assert first["messages"][0]["role"] == "user"
    assert first["messages"][1]["role"] == "assistant"

    dpo_lines = Path(data["dpo_path"]).read_text().strip().split("\n")
    assert len(dpo_lines) == 2
    dpo_entry = json.loads(dpo_lines[0])
    assert "prompt" in dpo_entry and "chosen" in dpo_entry and "rejected" in dpo_entry


def test_run_cycle_skip_training(tmp_path):
    mgr, fb_store, trigger, dist, trainer = _setup(tmp_path)
    collector = FeedbackCollector(node_id="test")
    for i in range(5):
        fb_store.save(collector.record_accepted(query=f"q{i}", response=f"r{i}"))

    result = trainer.run_cycle(skip_training=True)
    assert isinstance(result, TrainingCycleResult)
    assert result.cycle == 1
    assert result.sft_pairs == 5
    assert result.training_completed is True

    # Verify events were emitted
    events = mgr.events_since(None)
    learning_events = [e for e in events if e.category == "learning"]
    assert any(e.event_type == "training_started" for e in learning_events)
    assert any(e.event_type == "training_completed" for e in learning_events)

    # Trigger should be marked as used
    assert trigger.should_trigger() is False


def test_cycle_count_persists(tmp_path):
    _, fb_store, _, _, trainer = _setup(tmp_path)
    collector = FeedbackCollector(node_id="test")
    for i in range(5):
        fb_store.save(collector.record_accepted(query=f"q{i}", response=f"r{i}"))

    trainer.run_cycle(skip_training=True)
    assert trainer._cycle_count == 1

    # Create new trainer instance — should load persisted count
    _, _, trigger2, dist2, _ = _setup(tmp_path)
    trainer2 = AutoTrainer(
        feedback_store=fb_store, training_trigger=trigger2,
        model_distributor=dist2, mesh_manager=MeshManager(
            identity=NodeIdentity.generate(), data_dir=tmp_path / "mesh2"
        ),
        base_dir=tmp_path / "training",
    )
    assert trainer2._cycle_count == 1


def test_result_to_dict(tmp_path):
    result = TrainingCycleResult(
        cycle=1, sft_pairs=100, dpo_pairs=20,
        training_completed=True, model_name="PyMasters/Homie:v1",
        score=0.85, promoted=True,
    )
    d = result.to_dict()
    assert d["cycle"] == 1 and d["promoted"] is True and d["score"] == 0.85


def test_get_system_prompt(tmp_path):
    _, _, _, _, trainer = _setup(tmp_path)
    prompt = trainer._get_system_prompt()
    # Should extract from Modelfile if it exists
    if (Path(__file__).parent.parent.parent.parent / "Modelfile").exists():
        assert "Homie" in prompt or len(prompt) > 0
