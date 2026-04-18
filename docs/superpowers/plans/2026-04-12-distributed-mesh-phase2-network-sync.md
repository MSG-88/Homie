# Distributed Mesh — Phase 2: Network Transport & Event Sync

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Homie nodes can connect via WebSocket, exchange events, and sync state. Event log with ULID-based events, delta sync protocol, HMAC-authenticated transport, and offline resilience with replay on reconnect.

**Architecture:** Event log stored in SQLite per node. Each event has a ULID (time-sortable), category, type, and payload. Hub and Spokes exchange events via WebSocket with HMAC auth. Delta sync: Spoke sends last-seen event ID, Hub sends everything after. Vector clocks for causal ordering. Offline events accumulate locally and replay on reconnect.

**Tech Stack:** Python 3.11+, websockets, SQLite, HMAC-SHA256, ULID (custom lightweight impl to avoid dependency)

**Builds on Phase 1:** `mesh/identity.py`, `mesh/registry.py`, `mesh/pairing.py`, `mesh/capabilities.py`, `mesh/election.py`, `network/discovery.py`, `network/protocol.py`, `network/server.py`, `config.py` (MeshConfig)

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `src/homie_core/mesh/events.py` | Event dataclass, ULID generation, serialization |
| `src/homie_core/mesh/event_store.py` | SQLite event log: append, query since ID, compact |
| `src/homie_core/mesh/vector_clock.py` | Vector clock for causal event ordering |
| `src/homie_core/mesh/sync_protocol.py` | Delta sync logic: diff, merge, conflict resolution |
| `src/homie_core/mesh/transport.py` | WebSocket client/server with HMAC auth |
| `src/homie_core/mesh/mesh_manager.py` | Top-level coordinator: discovery + connect + sync loop |
| `tests/unit/test_mesh/test_events.py` | Event tests |
| `tests/unit/test_mesh/test_event_store.py` | Event store tests |
| `tests/unit/test_mesh/test_vector_clock.py` | Vector clock tests |
| `tests/unit/test_mesh/test_sync_protocol.py` | Sync protocol tests |
| `tests/unit/test_mesh/test_transport.py` | Transport auth tests |
| `tests/integration/test_mesh_sync.py` | End-to-end sync test |

---

### Task 1: ULID Generation and Event Data Model

**Files:**
- Create: `src/homie_core/mesh/events.py`
- Test: `tests/unit/test_mesh/test_events.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_events.py
import json
import time

from homie_core.mesh.events import HomieEvent, generate_ulid


def test_generate_ulid_format():
    """ULID is 26 chars, alphanumeric, time-sortable."""
    ulid = generate_ulid()
    assert len(ulid) == 26
    assert ulid.isalnum()


def test_ulid_time_sortable():
    """ULIDs generated in sequence sort chronologically."""
    a = generate_ulid()
    time.sleep(0.002)
    b = generate_ulid()
    assert a < b


def test_ulid_uniqueness():
    """1000 ULIDs are all unique."""
    ulids = {generate_ulid() for _ in range(1000)}
    assert len(ulids) == 1000


def test_event_creation():
    """HomieEvent holds all required fields."""
    evt = HomieEvent(
        node_id="node-1",
        category="memory",
        event_type="fact_learned",
        payload={"fact": "user likes Python"},
    )
    assert evt.event_id  # auto-generated ULID
    assert evt.node_id == "node-1"
    assert evt.category == "memory"
    assert evt.event_type == "fact_learned"
    assert evt.timestamp != ""
    assert evt.checksum != ""


def test_event_to_dict_and_from_dict():
    """Events round-trip through dict serialization."""
    evt = HomieEvent(
        node_id="node-1",
        category="task",
        event_type="task_created",
        payload={"task": "write sync"},
    )
    d = evt.to_dict()
    assert d["node_id"] == "node-1"
    assert d["category"] == "task"

    restored = HomieEvent.from_dict(d)
    assert restored.event_id == evt.event_id
    assert restored.payload == evt.payload
    assert restored.checksum == evt.checksum


def test_event_to_json():
    """Events serialize to JSON string."""
    evt = HomieEvent(
        node_id="n1", category="system", event_type="test",
        payload={"key": "val"},
    )
    j = evt.to_json()
    parsed = json.loads(j)
    assert parsed["node_id"] == "n1"


def test_event_checksum_integrity():
    """Checksum changes if payload changes."""
    evt = HomieEvent(
        node_id="n1", category="memory", event_type="fact_learned",
        payload={"fact": "original"},
    )
    cs1 = evt.checksum

    evt2 = HomieEvent(
        node_id="n1", category="memory", event_type="fact_learned",
        payload={"fact": "modified"},
    )
    assert evt2.checksum != cs1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/events.py
"""Mesh events — the atom of distributed sync.

Every meaningful action in Homie produces an immutable HomieEvent.
Events use ULIDs (time-sortable, globally unique) for ordering.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

from homie_core.utils import utc_now

# ---------------------------------------------------------------------------
# Lightweight ULID generation (Crockford Base32, no external dependency)
# ---------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_LAST_TS = 0
_LAST_RANDOM = 0


def generate_ulid() -> str:
    """Generate a ULID: 10-char timestamp + 16-char random (Crockford Base32).

    Time-sortable and globally unique. Monotonic within the same millisecond.
    """
    global _LAST_TS, _LAST_RANDOM

    ts_ms = int(time.time() * 1000)

    if ts_ms == _LAST_TS:
        _LAST_RANDOM += 1
    else:
        _LAST_TS = ts_ms
        _LAST_RANDOM = int.from_bytes(os.urandom(10), "big")

    # Encode timestamp (48 bits = 10 Crockford chars)
    ts_chars = []
    t = ts_ms
    for _ in range(10):
        ts_chars.append(_CROCKFORD[t & 0x1F])
        t >>= 5
    ts_part = "".join(reversed(ts_chars))

    # Encode random (80 bits = 16 Crockford chars)
    r = _LAST_RANDOM & ((1 << 80) - 1)
    rand_chars = []
    for _ in range(16):
        rand_chars.append(_CROCKFORD[r & 0x1F])
        r >>= 5
    rand_part = "".join(reversed(rand_chars))

    return ts_part + rand_part


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

def _compute_checksum(payload: dict) -> str:
    """SHA-256 of the JSON-serialized payload."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class HomieEvent:
    """An immutable event in the distributed event log."""

    node_id: str
    category: str       # memory | context | task | preference | learning | system
    event_type: str     # e.g., "fact_learned", "activity_changed"
    payload: dict
    event_id: str = field(default_factory=generate_ulid)
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())
    vector_clock: dict = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = _compute_checksum(self.payload)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "category": self.category,
            "event_type": self.event_type,
            "payload": self.payload,
            "vector_clock": self.vector_clock,
            "checksum": self.checksum,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> HomieEvent:
        return cls(
            event_id=d["event_id"],
            node_id=d["node_id"],
            timestamp=d["timestamp"],
            category=d["category"],
            event_type=d["event_type"],
            payload=d["payload"],
            vector_clock=d.get("vector_clock", {}),
            checksum=d.get("checksum", ""),
        )

    @classmethod
    def from_json(cls, raw: str) -> HomieEvent:
        return cls.from_dict(json.loads(raw))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_events.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/events.py tests/unit/test_mesh/test_events.py
git commit -m "feat(mesh): add ULID generation and HomieEvent data model for distributed sync"
```

---

### Task 2: Event Store — SQLite Append-Only Log

**Files:**
- Create: `src/homie_core/mesh/event_store.py`
- Test: `tests/unit/test_mesh/test_event_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_event_store.py
import time

from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore


def test_store_initialize(tmp_path):
    """EventStore creates the table on init."""
    store = EventStore(tmp_path / "events.db")
    store.initialize()


def test_append_and_get(tmp_path):
    """Can append an event and retrieve it by ID."""
    store = EventStore(tmp_path / "events.db")
    store.initialize()

    evt = HomieEvent(node_id="n1", category="memory", event_type="fact_learned",
                     payload={"fact": "test"})
    store.append(evt)

    loaded = store.get(evt.event_id)
    assert loaded is not None
    assert loaded.node_id == "n1"
    assert loaded.payload == {"fact": "test"}


def test_events_since(tmp_path):
    """events_since returns events after a given ID (exclusive)."""
    store = EventStore(tmp_path / "events.db")
    store.initialize()

    events = []
    for i in range(5):
        evt = HomieEvent(node_id="n1", category="memory", event_type="fact",
                         payload={"i": i})
        store.append(evt)
        events.append(evt)
        time.sleep(0.002)  # Ensure different ULIDs

    # Get events after the 2nd one — should return 3 events
    since = store.events_since(events[1].event_id, limit=100)
    assert len(since) == 3
    assert since[0].payload["i"] == 2


def test_events_since_none(tmp_path):
    """events_since with None returns all events."""
    store = EventStore(tmp_path / "events.db")
    store.initialize()

    for i in range(3):
        store.append(HomieEvent(node_id="n1", category="task", event_type="test",
                                payload={"i": i}))

    all_events = store.events_since(None, limit=100)
    assert len(all_events) == 3


def test_events_by_category(tmp_path):
    """Can filter events by category."""
    store = EventStore(tmp_path / "events.db")
    store.initialize()

    store.append(HomieEvent(node_id="n1", category="memory", event_type="fact",
                            payload={"x": 1}))
    store.append(HomieEvent(node_id="n1", category="task", event_type="created",
                            payload={"x": 2}))
    store.append(HomieEvent(node_id="n1", category="memory", event_type="episode",
                            payload={"x": 3}))

    memory_events = store.events_by_category("memory")
    assert len(memory_events) == 2


def test_count(tmp_path):
    """count() returns total events."""
    store = EventStore(tmp_path / "events.db")
    store.initialize()

    assert store.count() == 0
    for i in range(5):
        store.append(HomieEvent(node_id="n1", category="task", event_type="t",
                                payload={"i": i}))
    assert store.count() == 5


def test_last_event_id(tmp_path):
    """last_event_id returns the most recent event ID."""
    store = EventStore(tmp_path / "events.db")
    store.initialize()

    assert store.last_event_id() is None

    evt = HomieEvent(node_id="n1", category="task", event_type="t", payload={})
    store.append(evt)
    assert store.last_event_id() == evt.event_id


def test_mark_synced(tmp_path):
    """Can mark events as synced to hub."""
    store = EventStore(tmp_path / "events.db")
    store.initialize()

    evt = HomieEvent(node_id="n1", category="task", event_type="t", payload={})
    store.append(evt)

    unsynced = store.unsynced_events(limit=100)
    assert len(unsynced) == 1

    store.mark_synced([evt.event_id])

    unsynced = store.unsynced_events(limit=100)
    assert len(unsynced) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_event_store.py -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/event_store.py
"""Event store — SQLite append-only log for mesh sync events."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from homie_core.mesh.events import HomieEvent


class EventStore:
    """Append-only SQLite event log for distributed sync."""

    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                event_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                vector_clock_json TEXT NOT NULL DEFAULT '{}',
                checksum TEXT NOT NULL DEFAULT '',
                synced_to_hub BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_category ON event_log(category, timestamp)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_sync ON event_log(synced_to_hub, event_id)"
        )
        self._conn.commit()

    def append(self, event: HomieEvent) -> None:
        self._conn.execute("""
            INSERT OR IGNORE INTO event_log
                (event_id, node_id, timestamp, category, event_type,
                 payload_json, vector_clock_json, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id, event.node_id, event.timestamp,
            event.category, event.event_type,
            json.dumps(event.payload), json.dumps(event.vector_clock),
            event.checksum,
        ))
        self._conn.commit()

    def get(self, event_id: str) -> Optional[HomieEvent]:
        row = self._conn.execute(
            "SELECT * FROM event_log WHERE event_id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def events_since(self, after_event_id: Optional[str], limit: int = 1000) -> list[HomieEvent]:
        if after_event_id is None:
            rows = self._conn.execute(
                "SELECT * FROM event_log ORDER BY event_id ASC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM event_log WHERE event_id > ? ORDER BY event_id ASC LIMIT ?",
                (after_event_id, limit),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def events_by_category(self, category: str, limit: int = 1000) -> list[HomieEvent]:
        rows = self._conn.execute(
            "SELECT * FROM event_log WHERE category = ? ORDER BY event_id ASC LIMIT ?",
            (category, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def unsynced_events(self, limit: int = 1000) -> list[HomieEvent]:
        rows = self._conn.execute(
            "SELECT * FROM event_log WHERE synced_to_hub = 0 ORDER BY event_id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def mark_synced(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        placeholders = ",".join("?" for _ in event_ids)
        self._conn.execute(
            f"UPDATE event_log SET synced_to_hub = 1 WHERE event_id IN ({placeholders})",
            event_ids,
        )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM event_log").fetchone()
        return row[0]

    def last_event_id(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT event_id FROM event_log ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> HomieEvent:
        return HomieEvent(
            event_id=row["event_id"],
            node_id=row["node_id"],
            timestamp=row["timestamp"],
            category=row["category"],
            event_type=row["event_type"],
            payload=json.loads(row["payload_json"]),
            vector_clock=json.loads(row["vector_clock_json"]),
            checksum=row["checksum"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_event_store.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/event_store.py tests/unit/test_mesh/test_event_store.py
git commit -m "feat(mesh): add SQLite event store for append-only distributed event log"
```

---

### Task 3: Vector Clock

**Files:**
- Create: `src/homie_core/mesh/vector_clock.py`
- Test: `tests/unit/test_mesh/test_vector_clock.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_vector_clock.py
from homie_core.mesh.vector_clock import VectorClock


def test_initial_clock_is_empty():
    vc = VectorClock()
    assert vc.to_dict() == {}


def test_increment():
    vc = VectorClock()
    vc.increment("node-a")
    assert vc.get("node-a") == 1
    vc.increment("node-a")
    assert vc.get("node-a") == 2


def test_merge_takes_max():
    vc1 = VectorClock({"a": 3, "b": 1})
    vc2 = VectorClock({"a": 1, "b": 5, "c": 2})
    vc1.merge(vc2)
    assert vc1.get("a") == 3
    assert vc1.get("b") == 5
    assert vc1.get("c") == 2


def test_happens_before():
    vc1 = VectorClock({"a": 1, "b": 2})
    vc2 = VectorClock({"a": 2, "b": 3})
    assert vc1.happens_before(vc2) is True
    assert vc2.happens_before(vc1) is False


def test_concurrent():
    vc1 = VectorClock({"a": 3, "b": 1})
    vc2 = VectorClock({"a": 1, "b": 3})
    assert vc1.happens_before(vc2) is False
    assert vc2.happens_before(vc1) is False
    assert vc1.is_concurrent(vc2) is True


def test_from_dict():
    vc = VectorClock.from_dict({"x": 5, "y": 10})
    assert vc.get("x") == 5
    assert vc.get("y") == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_vector_clock.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/vector_clock.py
"""Vector clock for causal ordering of distributed events."""
from __future__ import annotations


class VectorClock:
    """Tracks causal ordering across nodes."""

    def __init__(self, clocks: dict[str, int] | None = None):
        self._clocks: dict[str, int] = dict(clocks) if clocks else {}

    def increment(self, node_id: str) -> int:
        self._clocks[node_id] = self._clocks.get(node_id, 0) + 1
        return self._clocks[node_id]

    def get(self, node_id: str) -> int:
        return self._clocks.get(node_id, 0)

    def merge(self, other: VectorClock) -> None:
        for node_id, count in other._clocks.items():
            self._clocks[node_id] = max(self._clocks.get(node_id, 0), count)

    def happens_before(self, other: VectorClock) -> bool:
        all_keys = set(self._clocks) | set(other._clocks)
        at_least_one_less = False
        for k in all_keys:
            mine = self._clocks.get(k, 0)
            theirs = other._clocks.get(k, 0)
            if mine > theirs:
                return False
            if mine < theirs:
                at_least_one_less = True
        return at_least_one_less

    def is_concurrent(self, other: VectorClock) -> bool:
        return not self.happens_before(other) and not other.happens_before(self)

    def to_dict(self) -> dict[str, int]:
        return dict(self._clocks)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> VectorClock:
        return cls(d)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_vector_clock.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/vector_clock.py tests/unit/test_mesh/test_vector_clock.py
git commit -m "feat(mesh): add vector clock for causal event ordering"
```

---

### Task 4: Delta Sync Protocol

**Files:**
- Create: `src/homie_core/mesh/sync_protocol.py`
- Test: `tests/unit/test_mesh/test_sync_protocol.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_sync_protocol.py
from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore
from homie_core.mesh.sync_protocol import SyncProtocol, SyncRequest, SyncResponse
import time


def _make_event(node_id: str, category: str, event_type: str, payload: dict) -> HomieEvent:
    evt = HomieEvent(node_id=node_id, category=category, event_type=event_type, payload=payload)
    time.sleep(0.002)
    return evt


def test_sync_request_creation():
    req = SyncRequest(node_id="spoke-1", last_event_id="01ABCDEF", vector_clock={"spoke-1": 5})
    assert req.node_id == "spoke-1"
    assert req.last_event_id == "01ABCDEF"


def test_hub_prepares_delta(tmp_path):
    """Hub returns events the Spoke hasn't seen."""
    hub_store = EventStore(tmp_path / "hub.db")
    hub_store.initialize()
    proto = SyncProtocol("hub-1", hub_store)

    e1 = _make_event("hub-1", "memory", "fact", {"i": 1})
    e2 = _make_event("hub-1", "memory", "fact", {"i": 2})
    e3 = _make_event("hub-1", "task", "created", {"i": 3})
    hub_store.append(e1)
    hub_store.append(e2)
    hub_store.append(e3)

    req = SyncRequest(node_id="spoke-1", last_event_id=e1.event_id, vector_clock={})
    resp = proto.prepare_response(req)
    assert len(resp.events) == 2
    assert resp.events[0].payload["i"] == 2
    assert resp.events[1].payload["i"] == 3


def test_hub_returns_all_if_no_last_event(tmp_path):
    """First sync — Spoke has never synced before."""
    hub_store = EventStore(tmp_path / "hub.db")
    hub_store.initialize()
    proto = SyncProtocol("hub-1", hub_store)

    for i in range(3):
        hub_store.append(_make_event("hub-1", "memory", "fact", {"i": i}))

    req = SyncRequest(node_id="spoke-1", last_event_id=None, vector_clock={})
    resp = proto.prepare_response(req)
    assert len(resp.events) == 3


def test_spoke_applies_events(tmp_path):
    """Spoke applies received events to its local store."""
    spoke_store = EventStore(tmp_path / "spoke.db")
    spoke_store.initialize()
    proto = SyncProtocol("spoke-1", spoke_store)

    events = [_make_event("hub-1", "memory", "fact", {"i": i}) for i in range(3)]
    resp = SyncResponse(events=events, hub_event_id="last-id")

    applied = proto.apply_response(resp)
    assert applied == 3
    assert spoke_store.count() == 3


def test_spoke_skips_duplicates(tmp_path):
    """Applying same events twice doesn't duplicate."""
    spoke_store = EventStore(tmp_path / "spoke.db")
    spoke_store.initialize()
    proto = SyncProtocol("spoke-1", spoke_store)

    events = [_make_event("hub-1", "memory", "fact", {"i": i}) for i in range(2)]
    resp = SyncResponse(events=events, hub_event_id="last")

    proto.apply_response(resp)
    proto.apply_response(resp)  # Same events again
    assert spoke_store.count() == 2


def test_spoke_prepares_unsynced(tmp_path):
    """Spoke sends its unsynced events to Hub."""
    spoke_store = EventStore(tmp_path / "spoke.db")
    spoke_store.initialize()
    proto = SyncProtocol("spoke-1", spoke_store)

    for i in range(3):
        spoke_store.append(_make_event("spoke-1", "preference", "set", {"i": i}))

    events_to_push = proto.get_unsynced_events()
    assert len(events_to_push) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_sync_protocol.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/sync_protocol.py
"""Delta sync protocol — exchange events between Hub and Spokes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore


@dataclass
class SyncRequest:
    """Spoke -> Hub: request events since last sync."""
    node_id: str
    last_event_id: Optional[str]
    vector_clock: dict = field(default_factory=dict)


@dataclass
class SyncResponse:
    """Hub -> Spoke: events the Spoke hasn't seen."""
    events: list[HomieEvent] = field(default_factory=list)
    hub_event_id: Optional[str] = None  # Hub's latest event ID


class SyncProtocol:
    """Handles delta sync logic for a single node."""

    def __init__(self, node_id: str, event_store: EventStore):
        self._node_id = node_id
        self._store = event_store

    def prepare_response(self, request: SyncRequest) -> SyncResponse:
        """Hub side: prepare events that the requesting Spoke hasn't seen."""
        events = self._store.events_since(request.last_event_id, limit=5000)
        hub_last = self._store.last_event_id()
        return SyncResponse(events=events, hub_event_id=hub_last)

    def apply_response(self, response: SyncResponse) -> int:
        """Spoke side: apply events received from Hub to local store.

        Returns the number of new events applied.
        """
        applied = 0
        for event in response.events:
            existing = self._store.get(event.event_id)
            if existing is None:
                self._store.append(event)
                applied += 1
        return applied

    def get_unsynced_events(self, limit: int = 5000) -> list[HomieEvent]:
        """Get local events that haven't been synced to Hub."""
        return self._store.unsynced_events(limit=limit)

    def mark_events_synced(self, event_ids: list[str]) -> None:
        """Mark events as successfully synced to Hub."""
        self._store.mark_synced(event_ids)

    def prepare_request(self) -> SyncRequest:
        """Spoke side: build a sync request with last known event ID."""
        return SyncRequest(
            node_id=self._node_id,
            last_event_id=self._store.last_event_id(),
            vector_clock={},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_sync_protocol.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/sync_protocol.py tests/unit/test_mesh/test_sync_protocol.py
git commit -m "feat(mesh): add delta sync protocol with request/response and deduplication"
```

---

### Task 5: HMAC-Authenticated WebSocket Transport

**Files:**
- Create: `src/homie_core/mesh/transport.py`
- Test: `tests/unit/test_mesh/test_transport.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_transport.py
import hmac
import hashlib
import json
import time

from homie_core.mesh.transport import (
    sign_message, verify_signature, MeshMessage, MeshMessageType,
)


def test_sign_message():
    """sign_message produces a hex HMAC-SHA256 signature."""
    sig = sign_message(b"secret", "hello world")
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA256 hex digest


def test_verify_valid_signature():
    """Valid signature passes verification."""
    secret = b"my-secret-key"
    payload = "test payload"
    sig = sign_message(secret, payload)
    assert verify_signature(secret, payload, sig) is True


def test_verify_invalid_signature():
    """Wrong signature fails verification."""
    secret = b"my-secret-key"
    assert verify_signature(secret, "data", "bad_signature") is False


def test_verify_wrong_secret():
    """Signature with different secret fails."""
    sig = sign_message(b"key-a", "data")
    assert verify_signature(b"key-b", "data", sig) is False


def test_mesh_message_types():
    """MeshMessageType has all required types."""
    assert MeshMessageType.SYNC_REQUEST == "sync_request"
    assert MeshMessageType.SYNC_RESPONSE == "sync_response"
    assert MeshMessageType.EVENT_PUSH == "event_push"
    assert MeshMessageType.HEARTBEAT == "heartbeat"
    assert MeshMessageType.INFERENCE_REQUEST == "inference_request"
    assert MeshMessageType.INFERENCE_RESPONSE == "inference_response"


def test_mesh_message_round_trip():
    """MeshMessage serializes/deserializes with signature."""
    secret = b"test-key"
    msg = MeshMessage.create(
        msg_type=MeshMessageType.HEARTBEAT,
        node_id="node-1",
        payload={"status": "online"},
        secret=secret,
    )
    raw = msg.to_json()
    parsed = MeshMessage.from_json(raw)
    assert parsed.msg_type == MeshMessageType.HEARTBEAT
    assert parsed.node_id == "node-1"
    assert parsed.payload == {"status": "online"}
    assert parsed.verify(secret) is True


def test_mesh_message_rejects_tampered():
    """Tampered message fails verification."""
    secret = b"key"
    msg = MeshMessage.create(
        msg_type=MeshMessageType.HEARTBEAT,
        node_id="n1",
        payload={"ok": True},
        secret=secret,
    )
    raw = msg.to_json()
    # Tamper with the JSON
    d = json.loads(raw)
    d["payload"]["ok"] = False
    tampered = json.dumps(d)
    parsed = MeshMessage.from_json(tampered)
    assert parsed.verify(secret) is False


def test_mesh_message_rejects_expired():
    """Message older than max_age fails verification."""
    secret = b"key"
    msg = MeshMessage.create(
        msg_type=MeshMessageType.HEARTBEAT,
        node_id="n1",
        payload={},
        secret=secret,
    )
    # Fake an old timestamp
    msg.timestamp = time.time() - 120
    raw = msg.to_json()
    parsed = MeshMessage.from_json(raw)
    assert parsed.verify(secret, max_age_seconds=60) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_transport.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/transport.py
"""HMAC-authenticated message transport for mesh communication."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Optional


def sign_message(secret: bytes, payload: str) -> str:
    """Compute HMAC-SHA256 signature of a payload string."""
    return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()


def verify_signature(secret: bytes, payload: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature. Timing-safe comparison."""
    expected = sign_message(secret, payload)
    return hmac.compare_digest(expected, signature)


class MeshMessageType:
    """Message types for mesh WebSocket protocol."""
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    EVENT_PUSH = "event_push"
    HEARTBEAT = "heartbeat"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"


@dataclass
class MeshMessage:
    """Authenticated message for mesh communication."""

    msg_type: str
    node_id: str
    payload: dict
    timestamp: float
    signature: str = ""

    @classmethod
    def create(
        cls,
        msg_type: str,
        node_id: str,
        payload: dict,
        secret: bytes,
    ) -> MeshMessage:
        """Create a signed message."""
        ts = time.time()
        msg = cls(msg_type=msg_type, node_id=node_id, payload=payload, timestamp=ts)
        signable = msg._signable_string()
        msg.signature = sign_message(secret, signable)
        return msg

    def verify(self, secret: bytes, max_age_seconds: float = 60.0) -> bool:
        """Verify signature and timestamp freshness."""
        if max_age_seconds > 0:
            age = time.time() - self.timestamp
            if age > max_age_seconds:
                return False
        signable = self._signable_string()
        return verify_signature(secret, signable, self.signature)

    def _signable_string(self) -> str:
        """Deterministic string for signing: type + node_id + payload + timestamp."""
        payload_str = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return f"{self.msg_type}:{self.node_id}:{payload_str}:{self.timestamp}"

    def to_json(self) -> str:
        return json.dumps({
            "msg_type": self.msg_type,
            "node_id": self.node_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "signature": self.signature,
        })

    @classmethod
    def from_json(cls, raw: str) -> MeshMessage:
        d = json.loads(raw)
        return cls(
            msg_type=d["msg_type"],
            node_id=d["node_id"],
            payload=d["payload"],
            timestamp=d["timestamp"],
            signature=d.get("signature", ""),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_transport.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/transport.py tests/unit/test_mesh/test_transport.py
git commit -m "feat(mesh): add HMAC-authenticated message transport for WebSocket mesh protocol"
```

---

### Task 6: Mesh Manager — Top-Level Coordinator

**Files:**
- Create: `src/homie_core/mesh/mesh_manager.py`
- Test: `tests/integration/test_mesh_sync.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_mesh_sync.py
"""Integration test: two-node sync via MeshManager."""
import time

from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore
from homie_core.mesh.sync_protocol import SyncProtocol, SyncRequest, SyncResponse
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.identity import NodeIdentity


def test_mesh_manager_initialization(tmp_path):
    """MeshManager initializes with identity and event store."""
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    assert mgr.node_id == identity.node_id
    assert mgr.event_count() == 0


def test_mesh_manager_emit_event(tmp_path):
    """emit() creates and stores an event."""
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)

    mgr.emit("memory", "fact_learned", {"fact": "user likes tea"})
    assert mgr.event_count() == 1

    events = mgr.events_since(None)
    assert events[0].category == "memory"
    assert events[0].payload["fact"] == "user likes tea"
    assert events[0].node_id == identity.node_id


def test_two_node_sync_simulation(tmp_path):
    """Simulate Hub and Spoke syncing events via protocol objects."""
    # Setup Hub
    hub_identity = NodeIdentity.generate()
    hub_mgr = MeshManager(identity=hub_identity, data_dir=tmp_path / "hub")

    # Setup Spoke
    spoke_identity = NodeIdentity.generate()
    spoke_mgr = MeshManager(identity=spoke_identity, data_dir=tmp_path / "spoke")

    # Hub emits 3 events
    for i in range(3):
        hub_mgr.emit("memory", "fact_learned", {"fact": f"fact-{i}"})
        time.sleep(0.002)

    # Spoke emits 2 events
    for i in range(2):
        spoke_mgr.emit("preference", "feedback", {"rating": i})
        time.sleep(0.002)

    # Spoke requests sync from Hub
    request = spoke_mgr.prepare_sync_request()
    assert request.last_event_id is not None  # Spoke has its own events

    # Hub prepares response (all Hub events after Spoke's last known)
    # Since Spoke has never seen Hub events, use None for last_event_id
    first_sync_request = SyncRequest(node_id=spoke_identity.node_id, last_event_id=None, vector_clock={})
    response = hub_mgr.handle_sync_request(first_sync_request)
    assert len(response.events) == 3

    # Spoke applies Hub events
    applied = spoke_mgr.apply_sync_response(response)
    assert applied == 3
    assert spoke_mgr.event_count() == 5  # 2 own + 3 from Hub

    # Hub receives Spoke events
    spoke_events = spoke_mgr.get_unsynced_for_hub()
    assert len(spoke_events) == 2  # Spoke's own 2 events (not yet synced)


def test_idempotent_sync(tmp_path):
    """Syncing the same events twice doesn't duplicate."""
    hub = MeshManager(identity=NodeIdentity.generate(), data_dir=tmp_path / "hub")
    spoke = MeshManager(identity=NodeIdentity.generate(), data_dir=tmp_path / "spoke")

    hub.emit("task", "created", {"task": "test"})

    req = SyncRequest(node_id=spoke.node_id, last_event_id=None, vector_clock={})
    resp = hub.handle_sync_request(req)

    spoke.apply_sync_response(resp)
    spoke.apply_sync_response(resp)  # Again
    assert spoke.event_count() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_mesh_sync.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/mesh_manager.py
"""Mesh manager — top-level coordinator for mesh operations.

Owns the event store and sync protocol. Provides a simple API for
emitting events, syncing with peers, and querying the event log.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore
from homie_core.mesh.sync_protocol import SyncProtocol, SyncRequest, SyncResponse
from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.vector_clock import VectorClock


class MeshManager:
    """Top-level coordinator for mesh operations on a single node."""

    def __init__(self, identity: NodeIdentity, data_dir: Path | str):
        self._identity = identity
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._event_store = EventStore(self._data_dir / "events.db")
        self._event_store.initialize()

        self._sync = SyncProtocol(identity.node_id, self._event_store)
        self._vector_clock = VectorClock()

    @property
    def node_id(self) -> str:
        return self._identity.node_id

    def emit(self, category: str, event_type: str, payload: dict) -> HomieEvent:
        """Create and store a new event from this node."""
        self._vector_clock.increment(self._identity.node_id)
        event = HomieEvent(
            node_id=self._identity.node_id,
            category=category,
            event_type=event_type,
            payload=payload,
            vector_clock=self._vector_clock.to_dict(),
        )
        self._event_store.append(event)
        return event

    def event_count(self) -> int:
        return self._event_store.count()

    def events_since(self, after_event_id: Optional[str], limit: int = 1000) -> list[HomieEvent]:
        return self._event_store.events_since(after_event_id, limit=limit)

    # --- Sync API ---

    def prepare_sync_request(self) -> SyncRequest:
        """Spoke side: build a sync request."""
        return self._sync.prepare_request()

    def handle_sync_request(self, request: SyncRequest) -> SyncResponse:
        """Hub side: prepare delta response for a Spoke."""
        return self._sync.prepare_response(request)

    def apply_sync_response(self, response: SyncResponse) -> int:
        """Spoke side: apply events received from Hub."""
        return self._sync.apply_response(response)

    def get_unsynced_for_hub(self, limit: int = 5000) -> list[HomieEvent]:
        """Spoke side: get events that haven't been pushed to Hub."""
        return self._sync.get_unsynced_events(limit=limit)

    def mark_pushed_to_hub(self, event_ids: list[str]) -> None:
        """Mark events as successfully pushed to Hub."""
        self._sync.mark_events_synced(event_ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_mesh_sync.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run all mesh tests for regression check**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/ tests/integration/test_mesh_smoke.py tests/integration/test_mesh_sync.py -v`
Expected: All tests PASS (Phase 1 + Phase 2)

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/mesh/mesh_manager.py tests/integration/test_mesh_sync.py
git commit -m "feat(mesh): add MeshManager coordinator with emit, sync request/response, and two-node sync"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | ULID + Event Model | `mesh/events.py` | 7 |
| 2 | Event Store (SQLite) | `mesh/event_store.py` | 8 |
| 3 | Vector Clock | `mesh/vector_clock.py` | 6 |
| 4 | Delta Sync Protocol | `mesh/sync_protocol.py` | 6 |
| 5 | HMAC Transport | `mesh/transport.py` | 8 |
| 6 | Mesh Manager | `mesh/mesh_manager.py` | 4 integration |

**Total: 6 tasks, 39 tests, 6 new source files, 6 new test files**

After Phase 2 completes, Homie nodes can:
- Emit events with globally unique time-sortable IDs (ULID)
- Store events in an append-only SQLite log
- Track causal ordering with vector clocks
- Exchange events via delta sync (only what's new)
- Authenticate all messages with HMAC-SHA256
- Detect and reject tampered or expired messages
- Perform full two-node sync (Hub ↔ Spoke)
- Handle duplicate events idempotently
