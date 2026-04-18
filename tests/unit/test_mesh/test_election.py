from homie_core.mesh.election import elect_hub, ElectionCandidate


def test_highest_score_wins():
    candidates = [
        ElectionCandidate(node_id="a", capability_score=100.0, created_at="2026-01-01T00:00:00"),
        ElectionCandidate(node_id="b", capability_score=290.0, created_at="2026-01-02T00:00:00"),
        ElectionCandidate(node_id="c", capability_score=50.0, created_at="2026-01-03T00:00:00"),
    ]
    winner = elect_hub(candidates)
    assert winner.node_id == "b"


def test_tiebreak_by_created_at():
    candidates = [
        ElectionCandidate(node_id="new", capability_score=100.0, created_at="2026-03-01T00:00:00"),
        ElectionCandidate(node_id="old", capability_score=100.0, created_at="2026-01-01T00:00:00"),
    ]
    winner = elect_hub(candidates)
    assert winner.node_id == "old"


def test_single_candidate():
    candidates = [
        ElectionCandidate(node_id="solo", capability_score=10.0, created_at="2026-01-01T00:00:00"),
    ]
    winner = elect_hub(candidates)
    assert winner.node_id == "solo"


def test_empty_candidates_returns_none():
    winner = elect_hub([])
    assert winner is None
