from homie_core.behavioral.base import BaseObserver


class DummyObserver(BaseObserver):
    def tick(self):
        return {"data": "test"}
    def get_profile_updates(self):
        return {"key": "value"}


def test_observer_record():
    obs = DummyObserver(name="test")
    obs.record({"type": "event"})
    assert len(obs.get_observations()) == 1
    assert obs.get_observations()[0]["observer"] == "test"


def test_observer_clear():
    obs = DummyObserver(name="test")
    obs.record({"type": "event"})
    obs.clear_observations()
    assert len(obs.get_observations()) == 0


def test_observer_enabled():
    obs = DummyObserver(name="test", enabled=False)
    assert obs.enabled is False
