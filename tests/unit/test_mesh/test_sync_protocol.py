import time
from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore
from homie_core.mesh.sync_protocol import SyncProtocol, SyncRequest, SyncResponse


def _make_event(node_id, category, event_type, payload):
    evt = HomieEvent(node_id=node_id, category=category, event_type=event_type, payload=payload)
    time.sleep(0.002)
    return evt


def test_sync_request_creation():
    req = SyncRequest(node_id="spoke-1", last_event_id="01ABCDEF", vector_clock={"spoke-1": 5})
    assert req.node_id == "spoke-1"


def test_hub_prepares_delta(tmp_path):
    hub_store = EventStore(tmp_path / "hub.db")
    hub_store.initialize()
    proto = SyncProtocol("hub-1", hub_store)
    e1 = _make_event("hub-1", "memory", "fact", {"i": 1})
    e2 = _make_event("hub-1", "memory", "fact", {"i": 2})
    e3 = _make_event("hub-1", "task", "created", {"i": 3})
    hub_store.append(e1); hub_store.append(e2); hub_store.append(e3)
    req = SyncRequest(node_id="spoke-1", last_event_id=e1.event_id, vector_clock={})
    resp = proto.prepare_response(req)
    assert len(resp.events) == 2
    assert resp.events[0].payload["i"] == 2


def test_hub_returns_all_if_no_last_event(tmp_path):
    hub_store = EventStore(tmp_path / "hub.db")
    hub_store.initialize()
    proto = SyncProtocol("hub-1", hub_store)
    for i in range(3):
        hub_store.append(_make_event("hub-1", "memory", "fact", {"i": i}))
    req = SyncRequest(node_id="spoke-1", last_event_id=None, vector_clock={})
    resp = proto.prepare_response(req)
    assert len(resp.events) == 3


def test_spoke_applies_events(tmp_path):
    spoke_store = EventStore(tmp_path / "spoke.db")
    spoke_store.initialize()
    proto = SyncProtocol("spoke-1", spoke_store)
    events = [_make_event("hub-1", "memory", "fact", {"i": i}) for i in range(3)]
    resp = SyncResponse(events=events, hub_event_id="last-id")
    assert proto.apply_response(resp) == 3
    assert spoke_store.count() == 3


def test_spoke_skips_duplicates(tmp_path):
    spoke_store = EventStore(tmp_path / "spoke.db")
    spoke_store.initialize()
    proto = SyncProtocol("spoke-1", spoke_store)
    events = [_make_event("hub-1", "memory", "fact", {"i": i}) for i in range(2)]
    resp = SyncResponse(events=events, hub_event_id="last")
    proto.apply_response(resp)
    proto.apply_response(resp)
    assert spoke_store.count() == 2


def test_spoke_prepares_unsynced(tmp_path):
    spoke_store = EventStore(tmp_path / "spoke.db")
    spoke_store.initialize()
    proto = SyncProtocol("spoke-1", spoke_store)
    for i in range(3):
        spoke_store.append(_make_event("spoke-1", "preference", "set", {"i": i}))
    assert len(proto.get_unsynced_events()) == 3
