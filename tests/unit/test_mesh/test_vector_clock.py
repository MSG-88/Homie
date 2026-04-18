from homie_core.mesh.vector_clock import VectorClock


def test_initial_clock_is_empty():
    vc = VectorClock()
    assert vc.to_dict() == {}


def test_increment():
    vc = VectorClock()
    vc.increment("node-a")
    assert vc.get("node-a") == 1
    vc.increment("node-a")
    assert vc.get("node-a") == 2


def test_merge_takes_max():
    vc1 = VectorClock({"a": 3, "b": 1})
    vc2 = VectorClock({"a": 1, "b": 5, "c": 2})
    vc1.merge(vc2)
    assert vc1.get("a") == 3
    assert vc1.get("b") == 5
    assert vc1.get("c") == 2


def test_happens_before():
    vc1 = VectorClock({"a": 1, "b": 2})
    vc2 = VectorClock({"a": 2, "b": 3})
    assert vc1.happens_before(vc2) is True
    assert vc2.happens_before(vc1) is False


def test_concurrent():
    vc1 = VectorClock({"a": 3, "b": 1})
    vc2 = VectorClock({"a": 1, "b": 3})
    assert vc1.happens_before(vc2) is False
    assert vc2.happens_before(vc1) is False
    assert vc1.is_concurrent(vc2) is True


def test_from_dict():
    vc = VectorClock.from_dict({"x": 5, "y": 10})
    assert vc.get("x") == 5
    assert vc.get("y") == 10
