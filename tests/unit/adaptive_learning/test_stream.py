import time
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.observation.stream import ObservationStream
from homie_core.adaptive_learning.observation.signals import (
    LearningSignal,
    SignalType,
    SignalCategory,
)


def _make_signal(signal_type=SignalType.EXPLICIT, category=SignalCategory.PREFERENCE):
    return LearningSignal(
        signal_type=signal_type,
        category=category,
        source="test",
        data={"key": "value"},
        context={},
    )


class TestObservationStream:
    def test_subscribe_and_emit(self):
        stream = ObservationStream()
        received = []
        stream.subscribe(lambda sig: received.append(sig))
        stream.emit(_make_signal())
        time.sleep(0.05)
        assert len(received) == 1

    def test_category_filter(self):
        stream = ObservationStream()
        prefs = []
        stream.subscribe(lambda sig: prefs.append(sig), category=SignalCategory.PREFERENCE)
        stream.emit(_make_signal(category=SignalCategory.PREFERENCE))
        stream.emit(_make_signal(category=SignalCategory.ENGAGEMENT))
        time.sleep(0.05)
        assert len(prefs) == 1

    def test_multiple_subscribers(self):
        stream = ObservationStream()
        r1, r2 = [], []
        stream.subscribe(lambda s: r1.append(s))
        stream.subscribe(lambda s: r2.append(s))
        stream.emit(_make_signal())
        time.sleep(0.05)
        assert len(r1) == 1
        assert len(r2) == 1

    def test_subscriber_exception_doesnt_crash(self):
        stream = ObservationStream()
        good = []
        stream.subscribe(lambda s: (_ for _ in ()).throw(RuntimeError("crash")))
        stream.subscribe(lambda s: good.append(s))
        stream.emit(_make_signal())
        time.sleep(0.05)
        assert len(good) == 1

    def test_signal_history(self):
        stream = ObservationStream(history_size=5)
        for i in range(7):
            stream.emit(_make_signal())
        time.sleep(0.05)
        assert len(stream.recent_signals) == 5

    def test_shutdown(self):
        stream = ObservationStream()
        stream.shutdown()
        stream.emit(_make_signal())  # should not crash
