# tests/unit/self_healing/test_event_bus.py
import threading
import time
import pytest
from homie_core.self_healing.event_bus import EventBus, HealthEvent


class TestHealthEvent:
    def test_event_creation(self):
        evt = HealthEvent(
            module="inference",
            event_type="probe_result",
            severity="info",
            details={"latency_ms": 42},
        )
        assert evt.module == "inference"
        assert evt.event_type == "probe_result"
        assert evt.severity == "info"
        assert evt.timestamp > 0

    def test_event_to_dict(self):
        evt = HealthEvent(module="storage", event_type="recovery", severity="warning", details={})
        d = evt.to_dict()
        assert d["module"] == "storage"
        assert "timestamp" in d


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("probe_result", lambda evt: received.append(evt))
        evt = HealthEvent(module="test", event_type="probe_result", severity="info", details={})
        bus.publish(evt)
        time.sleep(0.05)  # async delivery
        assert len(received) == 1
        assert received[0].module == "test"

    def test_multiple_subscribers(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe("anomaly", lambda e: r1.append(e))
        bus.subscribe("anomaly", lambda e: r2.append(e))
        bus.publish(HealthEvent(module="m", event_type="anomaly", severity="warning", details={}))
        time.sleep(0.05)
        assert len(r1) == 1
        assert len(r2) == 1

    def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.publish(HealthEvent(module="a", event_type="probe_result", severity="info", details={}))
        bus.publish(HealthEvent(module="b", event_type="recovery", severity="warning", details={}))
        time.sleep(0.05)
        assert len(received) == 2

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe("test", cb)
        bus.unsubscribe("test", cb)
        bus.publish(HealthEvent(module="m", event_type="test", severity="info", details={}))
        time.sleep(0.05)
        assert len(received) == 0

    def test_subscriber_exception_doesnt_crash_bus(self):
        bus = EventBus()
        good_received = []

        def bad_handler(e):
            raise RuntimeError("handler crash")

        bus.subscribe("test", bad_handler)
        bus.subscribe("test", lambda e: good_received.append(e))
        bus.publish(HealthEvent(module="m", event_type="test", severity="info", details={}))
        time.sleep(0.05)
        assert len(good_received) == 1

    def test_shutdown_stops_processing(self):
        bus = EventBus()
        bus.shutdown()
        # Should not hang or raise
        bus.publish(HealthEvent(module="m", event_type="test", severity="info", details={}))
