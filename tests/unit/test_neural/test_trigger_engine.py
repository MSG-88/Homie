"""Tests for the TriggerEngine module."""

import time
import pytest

from homie_core.neural.proactive.trigger_engine import ProactiveTask, TriggerEngine


@pytest.fixture
def engine():
    return TriggerEngine()


def _task(trigger_type="schedule", trigger_config=None, **kwargs):
    return ProactiveTask(
        id=kwargs.get("id", "test_task"),
        trigger_type=trigger_type,
        trigger_config=trigger_config or {},
        action=kwargs.get("action", "test_action"),
        domain=kwargs.get("domain", "general"),
        priority=kwargs.get("priority", 5),
        last_run=kwargs.get("last_run", None),
        enabled=kwargs.get("enabled", True),
    )


# ---------------------------------------------------------------------------
# register / unregister
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_task(self, engine):
        t = _task()
        engine.register_task(t)
        assert len(engine.tasks) == 1

    def test_unregister_task(self, engine):
        t = _task()
        engine.register_task(t)
        assert engine.unregister_task("test_task") is True
        assert len(engine.tasks) == 0

    def test_unregister_nonexistent(self, engine):
        assert engine.unregister_task("nope") is False


# ---------------------------------------------------------------------------
# check_triggers — schedule
# ---------------------------------------------------------------------------

class TestScheduleTrigger:
    def test_fires_when_never_run(self, engine):
        t = _task(trigger_config={"interval_seconds": 3600})
        engine.register_task(t)
        triggered = engine.check_triggers({"timestamp": time.time()})
        assert len(triggered) == 1

    def test_fires_after_interval(self, engine):
        now = time.time()
        t = _task(trigger_config={"interval_seconds": 60}, last_run=now - 120)
        engine.register_task(t)
        triggered = engine.check_triggers({"timestamp": now})
        assert len(triggered) == 1

    def test_does_not_fire_before_interval(self, engine):
        now = time.time()
        t = _task(trigger_config={"interval_seconds": 3600}, last_run=now - 10)
        engine.register_task(t)
        triggered = engine.check_triggers({"timestamp": now})
        assert len(triggered) == 0


# ---------------------------------------------------------------------------
# check_triggers — event
# ---------------------------------------------------------------------------

class TestEventTrigger:
    def test_fires_on_matching_event(self, engine):
        t = _task("event", {"event_type": "invoice_received"})
        engine.register_task(t)
        triggered = engine.check_triggers({
            "events": [{"type": "invoice_received"}],
        })
        assert len(triggered) == 1

    def test_does_not_fire_without_event(self, engine):
        t = _task("event", {"event_type": "invoice_received"})
        engine.register_task(t)
        triggered = engine.check_triggers({"events": []})
        assert len(triggered) == 0

    def test_fires_with_string_events(self, engine):
        t = _task("event", {"event_type": "alert"})
        engine.register_task(t)
        triggered = engine.check_triggers({"events": ["alert"]})
        assert len(triggered) == 1


# ---------------------------------------------------------------------------
# check_triggers — threshold
# ---------------------------------------------------------------------------

class TestThresholdTrigger:
    def test_gt_fires(self, engine):
        t = _task("threshold", {"metric": "score", "operator": "gt", "value": 0.8})
        engine.register_task(t)
        triggered = engine.check_triggers({"metrics": {"score": 0.9}})
        assert len(triggered) == 1

    def test_gt_does_not_fire(self, engine):
        t = _task("threshold", {"metric": "score", "operator": "gt", "value": 0.8})
        engine.register_task(t)
        triggered = engine.check_triggers({"metrics": {"score": 0.5}})
        assert len(triggered) == 0

    def test_lt_fires(self, engine):
        t = _task("threshold", {"metric": "balance", "operator": "lt", "value": 100})
        engine.register_task(t)
        triggered = engine.check_triggers({"metrics": {"balance": 50}})
        assert len(triggered) == 1


# ---------------------------------------------------------------------------
# check_triggers — pattern
# ---------------------------------------------------------------------------

class TestPatternTrigger:
    def test_fires_on_truthy_pattern(self, engine):
        t = _task("pattern", {"pattern_key": "important_email_detected"})
        engine.register_task(t)
        triggered = engine.check_triggers({
            "patterns": {"important_email_detected": True},
        })
        assert len(triggered) == 1

    def test_does_not_fire_on_falsy(self, engine):
        t = _task("pattern", {"pattern_key": "important_email_detected"})
        engine.register_task(t)
        triggered = engine.check_triggers({
            "patterns": {"important_email_detected": False},
        })
        assert len(triggered) == 0


# ---------------------------------------------------------------------------
# disabled tasks / priority ordering
# ---------------------------------------------------------------------------

class TestMisc:
    def test_disabled_task_not_triggered(self, engine):
        t = _task(trigger_config={"interval_seconds": 60}, enabled=False)
        engine.register_task(t)
        triggered = engine.check_triggers({"timestamp": time.time()})
        assert len(triggered) == 0

    def test_triggered_sorted_by_priority(self, engine):
        t1 = _task(id="low", trigger_config={"interval_seconds": 60}, priority=8)
        t2 = _task(id="high", trigger_config={"interval_seconds": 60}, priority=1)
        engine.register_task(t1)
        engine.register_task(t2)
        triggered = engine.check_triggers({"timestamp": time.time()})
        assert triggered[0].id == "high"
        assert triggered[1].id == "low"

    def test_get_default_tasks(self, engine):
        defaults = engine.get_default_tasks()
        assert len(defaults) >= 8
        domains = {t.domain for t in defaults}
        assert "finance" in domains
        assert "accounting" in domains
        assert "legal" in domains
