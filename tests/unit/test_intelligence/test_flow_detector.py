from homie_core.intelligence.flow_detector import FlowDetector


def test_record_activity():
    fd = FlowDetector(window_size=10)
    fd.record_activity("coding")
    assert len(fd._window) == 1
    assert list(fd._window) == ["coding"]


def test_entropy_single_activity():
    fd = FlowDetector(window_size=10)
    for _ in range(10):
        fd.record_activity("coding")
    entropy = fd.compute_entropy()
    assert entropy == 0.0


def test_entropy_uniform():
    fd = FlowDetector(window_size=10)
    activities = ["coding", "writing", "browsing", "researching", "testing"]
    for i in range(10):
        fd.record_activity(activities[i % len(activities)])
    entropy = fd.compute_entropy()
    assert entropy > 1.0


def test_flow_score_deep_focus():
    fd = FlowDetector(window_size=20)
    for _ in range(20):
        fd.record_activity("coding")
    score = fd.get_flow_score()
    assert score > 0.7


def test_flow_score_scattered():
    fd = FlowDetector(window_size=20)
    activities = ["coding", "email", "browsing", "slack", "docs"]
    for i in range(20):
        fd.record_activity(activities[i % len(activities)])
    score = fd.get_flow_score()
    assert score < 0.4


def test_is_in_flow():
    fd = FlowDetector(window_size=10, flow_threshold=0.7)
    for _ in range(10):
        fd.record_activity("coding")
    assert fd.is_in_flow()


def test_get_focus_report():
    fd = FlowDetector(window_size=10)
    for _ in range(5):
        fd.record_activity("coding")
    for _ in range(5):
        fd.record_activity("writing")
    report = fd.get_focus_report()
    assert "entropy" in report
    assert "flow_score" in report
    assert "dominant_activity" in report
    assert "switch_rate" in report


def test_switch_rate():
    fd = FlowDetector(window_size=10)
    fd.record_activity("coding")
    fd.record_activity("coding")
    fd.record_activity("writing")
    fd.record_activity("coding")
    rate = fd.get_switch_rate()
    assert abs(rate - 2 / 3) < 0.1


def test_empty_detector():
    fd = FlowDetector(window_size=10)
    assert fd.compute_entropy() == 0.0
    assert fd.get_flow_score() == 0.5
