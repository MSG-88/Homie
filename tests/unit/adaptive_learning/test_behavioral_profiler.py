# tests/unit/adaptive_learning/test_behavioral_profiler.py
import pytest
from homie_core.adaptive_learning.knowledge.behavioral_profiler import BehavioralProfiler


class TestBehavioralProfiler:
    def test_record_and_get_pattern(self):
        profiler = BehavioralProfiler()
        profiler.record_observation(hour=9, category="app", value="VSCode")
        profiler.record_observation(hour=9, category="app", value="VSCode")
        profiler.record_observation(hour=9, category="app", value="Chrome")
        pattern = profiler.get_pattern(hour=9, category="app")
        assert pattern["VSCode"] == 2
        assert pattern["Chrome"] == 1

    def test_predict_returns_most_frequent(self):
        profiler = BehavioralProfiler()
        for _ in range(5):
            profiler.record_observation(hour=14, category="activity", value="coding")
        for _ in range(2):
            profiler.record_observation(hour=14, category="activity", value="email")
        assert profiler.predict(hour=14, category="activity") == "coding"

    def test_predict_unknown_returns_none(self):
        profiler = BehavioralProfiler()
        assert profiler.predict(hour=3, category="app") is None

    def test_get_work_hours(self):
        profiler = BehavioralProfiler()
        for hour in [9, 10, 11, 14, 15, 16]:
            for _ in range(5):
                profiler.record_observation(hour=hour, category="activity", value="coding")
        work_hours = profiler.get_work_hours()
        assert 9 in work_hours
        assert 3 not in work_hours

    def test_get_daily_summary(self):
        profiler = BehavioralProfiler()
        profiler.record_observation(hour=9, category="app", value="VSCode")
        profiler.record_observation(hour=22, category="app", value="Netflix")
        summary = profiler.get_daily_summary()
        assert isinstance(summary, dict)
