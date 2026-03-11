import math
from homie_core.neural.behavioral_profile import BehavioralProfile


def _make_embedding(seed: float, dim: int = 8) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(dim)]


def test_observe_adds_sample():
    bp = BehavioralProfile(embed_dim=8)
    bp.observe(_make_embedding(1.0))
    assert bp.sample_count == 1


def test_compute_mean():
    bp = BehavioralProfile(embed_dim=4)
    bp.observe([1.0, 0.0, 0.0, 0.0])
    bp.observe([0.0, 1.0, 0.0, 0.0])
    mean = bp.get_mean_vector()
    assert abs(mean[0] - 0.5) < 1e-6
    assert abs(mean[1] - 0.5) < 1e-6


def test_covariance_matrix():
    bp = BehavioralProfile(embed_dim=2)
    bp.observe([1.0, 0.0])
    bp.observe([0.0, 1.0])
    cov = bp._compute_covariance()
    assert len(cov) == 2
    assert len(cov[0]) == 2


def test_power_iteration_finds_eigenvector():
    bp = BehavioralProfile(embed_dim=4)
    for _ in range(20):
        bp.observe([1.0, 0.1, 0.0, 0.0])
        bp.observe([0.9, 0.05, 0.0, 0.0])
    eigenvecs = bp.compute_eigenvectors(top_k=1)
    assert len(eigenvecs) == 1
    assert abs(eigenvecs[0][0]) > 0.5


def test_behavioral_fingerprint():
    bp = BehavioralProfile(embed_dim=4)
    for _ in range(20):
        bp.observe([1.0, 0.5, 0.0, 0.0])
    fingerprint = bp.get_fingerprint(top_k=2)
    assert "eigenvectors" in fingerprint
    assert "explained_variance" in fingerprint
    assert "sample_count" in fingerprint


def test_compare_profiles_identical():
    bp1 = BehavioralProfile(embed_dim=4)
    bp2 = BehavioralProfile(embed_dim=4)
    for _ in range(20):
        bp1.observe([1.0, 0.5, 0.0, 0.0])
        bp2.observe([1.0, 0.5, 0.0, 0.0])
    similarity = BehavioralProfile.compare(bp1, bp2, top_k=2)
    assert similarity > 0.8


def test_compare_profiles_different():
    bp1 = BehavioralProfile(embed_dim=4)
    bp2 = BehavioralProfile(embed_dim=4)
    for _ in range(20):
        bp1.observe([1.0, 0.0, 0.0, 0.0])
        bp2.observe([0.0, 0.0, 0.0, 1.0])
    similarity = BehavioralProfile.compare(bp1, bp2, top_k=1)
    assert similarity < 0.5


def test_serialize_deserialize():
    bp = BehavioralProfile(embed_dim=4)
    for _ in range(5):
        bp.observe([1.0, 0.5, 0.2, 0.1])
    data = bp.serialize()
    bp2 = BehavioralProfile.deserialize(data)
    assert bp2.sample_count == 5
    assert bp2.get_mean_vector() == bp.get_mean_vector()
