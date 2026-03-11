import math
from homie_core.neural.rhythm_model import CircadianRhythmModel


def test_record_activity():
    model = CircadianRhythmModel()
    model.record_activity(hour=9, productivity_score=0.8, activity_type="coding")
    assert len(model._hourly_buckets[9]) == 1


def test_record_clamps_hour():
    model = CircadianRhythmModel()
    model.record_activity(hour=25, productivity_score=0.5)
    assert len(model._hourly_buckets[1]) == 1


def test_get_hourly_averages():
    model = CircadianRhythmModel()
    model.record_activity(hour=9, productivity_score=0.8)
    model.record_activity(hour=9, productivity_score=0.6)
    model.record_activity(hour=14, productivity_score=0.9)
    avgs = model.get_hourly_averages()
    assert abs(avgs[9] - 0.7) < 1e-6
    assert abs(avgs[14] - 0.9) < 1e-6
    assert avgs[0] == 0.0


def test_fourier_decompose_returns_components():
    model = CircadianRhythmModel()
    for day in range(7):
        for h in range(24):
            score = 0.5 + 0.4 * math.cos(2 * math.pi * (h - 10) / 24)
            model.record_activity(hour=h, productivity_score=max(0, min(1, score)))
    components = model.fourier_decompose(top_k=3)
    assert len(components) <= 3
    assert all("frequency" in c and "amplitude" in c and "phase" in c for c in components)
    assert components[0]["amplitude"] > 0.1


def test_predict_productivity():
    model = CircadianRhythmModel()
    for day in range(7):
        for h in range(24):
            score = 0.5 + 0.4 * math.cos(2 * math.pi * (h - 10) / 24)
            model.record_activity(hour=h, productivity_score=max(0, min(1, score)))
    pred_10 = model.predict_productivity(hour=10)
    pred_3 = model.predict_productivity(hour=3)
    assert pred_10 > pred_3


def test_get_optimal_windows():
    model = CircadianRhythmModel()
    for day in range(7):
        for h in range(24):
            score = 0.5 + 0.4 * math.cos(2 * math.pi * (h - 10) / 24)
            model.record_activity(hour=h, productivity_score=max(0, min(1, score)))
    windows = model.get_optimal_windows(top_n=3)
    assert len(windows) <= 3
    assert all("hour" in w and "predicted_score" in w for w in windows)
    assert abs(windows[0]["hour"] - 10) <= 2


def test_get_activity_rhythm():
    model = CircadianRhythmModel()
    model.record_activity(hour=9, productivity_score=0.8, activity_type="coding")
    model.record_activity(hour=9, productivity_score=0.7, activity_type="coding")
    model.record_activity(hour=14, productivity_score=0.6, activity_type="writing")
    rhythm = model.get_activity_rhythm()
    assert "coding" in rhythm
    assert 9 in rhythm["coding"]


def test_serialize_deserialize():
    model = CircadianRhythmModel()
    model.record_activity(hour=9, productivity_score=0.8, activity_type="coding")
    data = model.serialize()
    model2 = CircadianRhythmModel.deserialize(data)
    assert len(model2._hourly_buckets[9]) == 1
