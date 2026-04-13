from homie_core.mesh.feedback_collector import FeedbackCollector, FeedbackSignal, SignalType


def test_signal_types():
    assert SignalType.ACCEPTED == "accepted" and SignalType.REGENERATED == "regenerated"
    assert SignalType.CORRECTED == "corrected" and SignalType.IGNORED == "ignored" and SignalType.RATED == "rated"


def test_create_signal():
    sig = FeedbackSignal(signal_type=SignalType.ACCEPTED, query="What is Python?",
                         response_preview="Python is...", node_id="desktop", activity_context="coding")
    assert sig.signal_id and sig.signal_type == "accepted" and sig.timestamp != ""


def test_signal_to_dict_roundtrip():
    sig = FeedbackSignal(signal_type=SignalType.CORRECTED, query="fix this", response_preview="here's the fix",
                         node_id="laptop", activity_context="debugging")
    r = FeedbackSignal.from_dict(sig.to_dict())
    assert r.signal_id == sig.signal_id and r.signal_type == "corrected"


def test_collector_record_accepted():
    c = FeedbackCollector(node_id="desktop")
    sig = c.record_accepted(query="hello", response="Hi there!")
    assert sig.signal_type == SignalType.ACCEPTED and sig.response_preview == "Hi there!" and len(c.signals) == 1


def test_collector_record_regenerated():
    c = FeedbackCollector(node_id="desktop")
    assert c.record_regenerated(query="explain X", original="bad", regenerated="better").signal_type == SignalType.REGENERATED


def test_collector_record_corrected():
    c = FeedbackCollector(node_id="desktop")
    sig = c.record_corrected(query="what is 2+2?", original="5", correction="4")
    assert sig.signal_type == SignalType.CORRECTED and "correction" in sig.metadata


def test_collector_record_ignored():
    assert FeedbackCollector(node_id="d").record_ignored(query="stuff", response="...").signal_type == SignalType.IGNORED


def test_collector_flush():
    c = FeedbackCollector(node_id="desktop")
    c.record_accepted(query="q1", response="r1"); c.record_accepted(query="q2", response="r2")
    assert len(c.flush()) == 2 and len(c.signals) == 0
