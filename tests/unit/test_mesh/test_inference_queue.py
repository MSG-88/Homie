import time, threading
from homie_core.mesh.inference_queue import InferenceQueue, InferenceRequest, InferencePriority


def test_priority_levels():
    assert InferencePriority.IMMEDIATE < InferencePriority.BACKGROUND
    assert InferencePriority.BACKGROUND < InferencePriority.BATCH


def test_submit_and_get():
    q = InferenceQueue(max_concurrent=2)
    req = InferenceRequest(request_id="r1", node_id="spoke-1", prompt="hello",
                           max_tokens=100, temperature=0.7, priority=InferencePriority.IMMEDIATE)
    q.submit(req)
    assert q.pending_count() == 1
    got = q.get(timeout=1.0)
    assert got is not None and got.request_id == "r1"


def test_priority_ordering():
    q = InferenceQueue(max_concurrent=5)
    q.submit(InferenceRequest(request_id="batch", node_id="n1", prompt="p", max_tokens=100, temperature=0.7, priority=InferencePriority.BATCH))
    q.submit(InferenceRequest(request_id="immediate", node_id="n1", prompt="p", max_tokens=100, temperature=0.7, priority=InferencePriority.IMMEDIATE))
    q.submit(InferenceRequest(request_id="background", node_id="n1", prompt="p", max_tokens=100, temperature=0.7, priority=InferencePriority.BACKGROUND))
    order = [q.get(timeout=1.0).request_id for _ in range(3)]
    assert order == ["immediate", "background", "batch"]


def test_max_concurrent_blocking():
    q = InferenceQueue(max_concurrent=1)
    q.submit(InferenceRequest(request_id="r1", node_id="n1", prompt="p", max_tokens=100, temperature=0.7, priority=InferencePriority.IMMEDIATE))
    got = q.get(timeout=1.0)
    q.mark_active(got.request_id)
    assert q.active_count() == 1 and q.can_accept() is False
    q.mark_done(got.request_id)
    assert q.active_count() == 0 and q.can_accept() is True


def test_get_timeout_returns_none():
    q = InferenceQueue(max_concurrent=2)
    assert q.get(timeout=0.1) is None


def test_queue_stats():
    q = InferenceQueue(max_concurrent=5)
    for i in range(3):
        q.submit(InferenceRequest(request_id=f"r{i}", node_id="n1", prompt="p", max_tokens=100, temperature=0.7, priority=InferencePriority.IMMEDIATE))
    stats = q.stats()
    assert stats["pending"] == 3 and stats["active"] == 0
    req = q.get(timeout=1.0)
    q.mark_active(req.request_id)
    q.mark_done(req.request_id)
    assert q.stats()["completed"] == 1
