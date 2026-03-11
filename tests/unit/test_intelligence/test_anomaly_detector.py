import random
from homie_core.intelligence.anomaly_detector import AnomalyDetector


def test_fit_and_score():
    ad = AnomalyDetector(n_trees=10, sample_size=32)
    data = [[float(i % 5), float(i % 3)] for i in range(50)]
    ad.fit(data)
    score = ad.score([2.0, 1.0])
    assert 0.0 <= score <= 1.0


def test_anomaly_scores_higher():
    ad = AnomalyDetector(n_trees=50, sample_size=32)
    random.seed(42)
    normal = [[random.gauss(0, 0.5), random.gauss(0, 0.5)] for _ in range(100)]
    ad.fit(normal)
    normal_score = ad.score([0.1, -0.1])
    anomaly_score = ad.score([10.0, 10.0])
    assert anomaly_score > normal_score


def test_is_anomaly():
    ad = AnomalyDetector(n_trees=50, sample_size=32, threshold=0.6)
    random.seed(42)
    normal = [[random.gauss(0, 0.5), random.gauss(0, 0.5)] for _ in range(100)]
    ad.fit(normal)
    assert not ad.is_anomaly([0.0, 0.0])
    assert ad.is_anomaly([20.0, 20.0])


def test_streaming_update():
    ad = AnomalyDetector(n_trees=10, sample_size=16)
    data = [[float(i), float(i)] for i in range(20)]
    ad.fit(data)
    ad.stream_update([100.0, 100.0])
    score = ad.score([0.0, 0.0])
    assert 0.0 <= score <= 1.0


def test_empty_forest_returns_neutral():
    ad = AnomalyDetector(n_trees=10, sample_size=16)
    score = ad.score([1.0, 2.0])
    assert score == 0.5


def test_single_dimension():
    ad = AnomalyDetector(n_trees=20, sample_size=16)
    data = [[float(i)] for i in range(30)]
    ad.fit(data)
    score = ad.score([15.0])
    assert 0.0 <= score <= 1.0


def test_get_feature_importance():
    ad = AnomalyDetector(n_trees=30, sample_size=32)
    random.seed(42)
    data = [[random.gauss(0, 5), 1.0] for _ in range(50)]
    ad.fit(data)
    importance = ad.get_feature_importance()
    assert len(importance) == 2
    assert importance[0] >= importance[1]


def test_serialize_deserialize():
    ad = AnomalyDetector(n_trees=10, sample_size=16)
    data = [[float(i), float(i * 2)] for i in range(20)]
    ad.fit(data)
    score_before = ad.score([5.0, 10.0])
    state = ad.serialize()
    ad2 = AnomalyDetector.deserialize(state)
    score_after = ad2.score([5.0, 10.0])
    assert abs(score_before - score_after) < 1e-6
