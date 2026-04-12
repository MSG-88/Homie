import time
from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore

def test_store_initialize(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()

def test_append_and_get(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    evt = HomieEvent(node_id="n1", category="memory", event_type="fact_learned", payload={"fact": "test"})
    store.append(evt)
    loaded = store.get(evt.event_id)
    assert loaded is not None
    assert loaded.node_id == "n1"
    assert loaded.payload == {"fact": "test"}

def test_events_since(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    events = []
    for i in range(5):
        evt = HomieEvent(node_id="n1", category="memory", event_type="fact", payload={"i": i})
        store.append(evt)
        events.append(evt)
        time.sleep(0.002)
    since = store.events_since(events[1].event_id, limit=100)
    assert len(since) == 3
    assert since[0].payload["i"] == 2

def test_events_since_none(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    for i in range(3):
        store.append(HomieEvent(node_id="n1", category="task", event_type="test", payload={"i": i}))
    all_events = store.events_since(None, limit=100)
    assert len(all_events) == 3

def test_events_by_category(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    store.append(HomieEvent(node_id="n1", category="memory", event_type="fact", payload={"x": 1}))
    store.append(HomieEvent(node_id="n1", category="task", event_type="created", payload={"x": 2}))
    store.append(HomieEvent(node_id="n1", category="memory", event_type="episode", payload={"x": 3}))
    assert len(store.events_by_category("memory")) == 2

def test_count(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    assert store.count() == 0
    for i in range(5):
        store.append(HomieEvent(node_id="n1", category="task", event_type="t", payload={"i": i}))
    assert store.count() == 5

def test_last_event_id(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    assert store.last_event_id() is None
    evt = HomieEvent(node_id="n1", category="task", event_type="t", payload={})
    store.append(evt)
    assert store.last_event_id() == evt.event_id

def test_mark_synced(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    evt = HomieEvent(node_id="n1", category="task", event_type="t", payload={})
    store.append(evt)
    assert len(store.unsynced_events(limit=100)) == 1
    store.mark_synced([evt.event_id])
    assert len(store.unsynced_events(limit=100)) == 0
