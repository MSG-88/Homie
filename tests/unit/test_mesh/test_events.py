import json, time
from homie_core.mesh.events import HomieEvent, generate_ulid

def test_generate_ulid_format():
    ulid = generate_ulid()
    assert len(ulid) == 26
    assert ulid.isalnum()

def test_ulid_time_sortable():
    a = generate_ulid()
    time.sleep(0.002)
    b = generate_ulid()
    assert a < b

def test_ulid_uniqueness():
    ulids = {generate_ulid() for _ in range(1000)}
    assert len(ulids) == 1000

def test_event_creation():
    evt = HomieEvent(node_id="node-1", category="memory", event_type="fact_learned",
                     payload={"fact": "user likes Python"})
    assert evt.event_id
    assert evt.node_id == "node-1"
    assert evt.category == "memory"
    assert evt.timestamp != ""
    assert evt.checksum != ""

def test_event_to_dict_and_from_dict():
    evt = HomieEvent(node_id="node-1", category="task", event_type="task_created",
                     payload={"task": "write sync"})
    d = evt.to_dict()
    restored = HomieEvent.from_dict(d)
    assert restored.event_id == evt.event_id
    assert restored.payload == evt.payload
    assert restored.checksum == evt.checksum

def test_event_to_json():
    evt = HomieEvent(node_id="n1", category="system", event_type="test", payload={"key": "val"})
    parsed = json.loads(evt.to_json())
    assert parsed["node_id"] == "n1"

def test_event_checksum_integrity():
    evt1 = HomieEvent(node_id="n1", category="memory", event_type="fact_learned",
                      payload={"fact": "original"})
    evt2 = HomieEvent(node_id="n1", category="memory", event_type="fact_learned",
                      payload={"fact": "modified"})
    assert evt2.checksum != evt1.checksum
