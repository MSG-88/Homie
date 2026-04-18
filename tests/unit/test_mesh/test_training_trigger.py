"""Unit tests for TrainingTrigger."""
from __future__ import annotations

from homie_core.mesh.feedback_collector import FeedbackSignal, SignalType
from homie_core.mesh.feedback_store import FeedbackStore
from homie_core.mesh.training_trigger import TrainingTrigger


def test_not_ready_when_empty(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db")
    s.initialize()
    assert TrainingTrigger(feedback_store=s).should_trigger() is False


def test_triggers_at_signal_threshold(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db")
    s.initialize()
    t = TrainingTrigger(feedback_store=s, min_signals=10)
    for i in range(10):
        s.save(
            FeedbackSignal(
                signal_type=SignalType.ACCEPTED,
                query=f"q{i}",
                response_preview="r",
                node_id="n1",
            )
        )
    assert t.should_trigger() is True


def test_not_ready_below_threshold(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db")
    s.initialize()
    t = TrainingTrigger(feedback_store=s, min_signals=100)
    for i in range(5):
        s.save(
            FeedbackSignal(
                signal_type=SignalType.ACCEPTED,
                query=f"q{i}",
                response_preview="r",
                node_id="n1",
            )
        )
    assert t.should_trigger() is False


def test_triggers_at_correction_threshold(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db")
    s.initialize()
    t = TrainingTrigger(feedback_store=s, min_signals=1000, min_corrections=5)
    for i in range(5):
        s.save(
            FeedbackSignal(
                signal_type=SignalType.CORRECTED,
                query=f"q{i}",
                response_preview="bad",
                node_id="n1",
                metadata={"correction": "good"},
            )
        )
    assert t.should_trigger() is True


def test_get_training_summary(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db")
    s.initialize()
    for i in range(3):
        s.save(
            FeedbackSignal(
                signal_type=SignalType.ACCEPTED,
                query=f"q{i}",
                response_preview="r",
                node_id="n1",
            )
        )
    s.save(
        FeedbackSignal(
            signal_type=SignalType.CORRECTED,
            query="fix",
            response_preview="bad",
            node_id="n1",
            metadata={"correction": "good"},
        )
    )
    t = TrainingTrigger(feedback_store=s, min_signals=2)
    summary = t.get_summary()
    assert (
        summary["total_signals"] == 4
        and summary["sft_pairs"] >= 3
        and summary["dpo_pairs"] >= 1
        and summary["ready"] is True
    )


def test_mark_triggered(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db")
    s.initialize()
    t = TrainingTrigger(feedback_store=s, min_signals=3)
    for i in range(3):
        s.save(
            FeedbackSignal(
                signal_type=SignalType.ACCEPTED,
                query=f"q{i}",
                response_preview="r",
                node_id="n1",
            )
        )
    assert t.should_trigger() is True
    t.mark_triggered()
    assert t.should_trigger() is False
