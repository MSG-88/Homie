"""Integration tests for the full self-learning loop across mesh nodes."""
from __future__ import annotations

import time

from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.feedback_collector import FeedbackCollector, FeedbackSignal, SignalType
from homie_core.mesh.feedback_store import FeedbackStore
from homie_core.mesh.training_trigger import TrainingTrigger
from homie_core.mesh.model_distributor import ModelDistributor
from homie_core.mesh.sync_protocol import SyncRequest


def test_full_learning_loop(tmp_path):
    # --- Node setup ---
    hub_id = NodeIdentity.generate()
    hub_mgr = MeshManager(identity=hub_id, data_dir=tmp_path / "hub")
    spoke_id = NodeIdentity.generate()
    spoke_mgr = MeshManager(identity=spoke_id, data_dir=tmp_path / "spoke")

    # --- Collect feedback on spoke ---
    collector = FeedbackCollector(node_id=spoke_id.node_id)
    collector.record_accepted(query="What is Python?", response="A programming language")
    collector.record_corrected(query="2+2?", original="5", correction="4")
    collector.record_accepted(query="Hello", response="Hi there!")

    spoke_store = FeedbackStore(tmp_path / "spoke" / "feedback.db")
    spoke_store.initialize()
    for sig in collector.flush():
        spoke_store.save(sig)
        spoke_mgr.emit("preference", "feedback_signal", sig.to_dict())

    assert spoke_mgr.event_count() == 3

    # --- Sync spoke → hub ---
    req = SyncRequest(node_id=hub_id.node_id, last_event_id=None, vector_clock={})
    resp = spoke_mgr.handle_sync_request(req)
    hub_mgr.apply_sync_response(resp)
    assert hub_mgr.event_count() == 3

    # --- Populate hub feedback store from synced events ---
    hub_store = FeedbackStore(tmp_path / "hub" / "feedback.db")
    hub_store.initialize()
    for evt in hub_mgr.events_since(None):
        if evt.event_type == "feedback_signal":
            hub_store.save(FeedbackSignal.from_dict(evt.payload))

    # --- Training trigger ---
    trigger = TrainingTrigger(feedback_store=hub_store, min_signals=3)
    assert trigger.should_trigger() is True

    # --- Distribute model lifecycle events from hub ---
    dist = ModelDistributor(mesh_manager=hub_mgr)
    dist.announce_training_started(cycle=1, sft_pairs=2, dpo_pairs=1)
    dist.announce_training_completed(cycle=1, score=0.85, promoted=True)
    dist.announce_update(
        model_name="homie-v2",
        model_path="/models/v2.gguf",
        score_improvement=0.05,
        cycle=1,
    )
    assert len(dist.get_model_history()) == 1

    # --- Sync hub → spoke ---
    req2 = SyncRequest(node_id=spoke_id.node_id, last_event_id=None, vector_clock={})
    resp2 = hub_mgr.handle_sync_request(req2)
    spoke_mgr.apply_sync_response(resp2)

    learning_events = [e for e in spoke_mgr.events_since(None) if e.category == "learning"]
    assert len(learning_events) == 3
    assert any(e.event_type == "model_updated" for e in learning_events)

    # --- Mark triggered and verify cooldown ---
    trigger.mark_triggered()
    assert trigger.should_trigger() is False


def test_training_pairs_extraction(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    store.initialize()
    collector = FeedbackCollector(node_id="test")

    for i in range(5):
        store.save(collector.record_accepted(query=f"q{i}", response=f"good {i}"))
    for i in range(3):
        store.save(
            collector.record_corrected(query=f"fix{i}", original=f"bad{i}", correction=f"good{i}")
        )

    pairs = store.get_training_pairs()
    sft = [p for p in pairs if p["type"] == "sft"]
    dpo = [p for p in pairs if p["type"] == "dpo"]

    assert len(sft) == 5 and len(dpo) == 3
    assert all(p["response"].startswith("good") for p in sft)
    assert all(p["chosen"].startswith("good") for p in dpo)
