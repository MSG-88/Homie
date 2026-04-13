"""Tests for FeedbackStore — SQLite persistence for learning signals."""
import time

import pytest

from homie_core.mesh.feedback_collector import FeedbackSignal, SignalType
from homie_core.mesh.feedback_store import FeedbackStore


def test_store_initialize(tmp_path):
    FeedbackStore(tmp_path / "feedback.db").initialize()


def test_save_and_get(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db"); s.initialize()
    sig = FeedbackSignal(signal_type=SignalType.ACCEPTED, query="hi", response_preview="hello", node_id="n1")
    s.save(sig)
    loaded = s.get(sig.signal_id)
    assert loaded and loaded.signal_type == "accepted" and loaded.query == "hi"


def test_count_by_type(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db"); s.initialize()
    for _ in range(3): s.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query="q", response_preview="r", node_id="n1"))
    for _ in range(2): s.save(FeedbackSignal(signal_type=SignalType.CORRECTED, query="q", response_preview="r", node_id="n1"))
    c = s.count_by_type()
    assert c["accepted"] == 3 and c["corrected"] == 2


def test_total_count(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db"); s.initialize()
    assert s.total_count() == 0
    for i in range(5): s.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query=f"q{i}", response_preview="r", node_id="n1"))
    assert s.total_count() == 5


def test_signals_since(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db"); s.initialize()
    sigs = []
    for i in range(4):
        sig = FeedbackSignal(signal_type=SignalType.ACCEPTED, query=f"q{i}", response_preview="r", node_id="n1")
        s.save(sig); sigs.append(sig); time.sleep(0.002)
    assert len(s.signals_since(sigs[1].signal_id, limit=100)) == 2


def test_get_training_pairs(tmp_path):
    s = FeedbackStore(tmp_path / "fb.db"); s.initialize()
    s.save(FeedbackSignal(signal_type=SignalType.CORRECTED, query="what is 2+2?", response_preview="5", node_id="n1", metadata={"correction": "4"}))
    s.save(FeedbackSignal(signal_type=SignalType.ACCEPTED, query="hello", response_preview="Hi!", node_id="n1"))
    pairs = s.get_training_pairs()
    dpo = [p for p in pairs if p["type"] == "dpo"]
    sft = [p for p in pairs if p["type"] == "sft"]
    assert len(dpo) == 1 and dpo[0]["rejected"] == "5" and dpo[0]["chosen"] == "4"
    assert len(sft) == 1 and sft[0]["response"] == "Hi!"
