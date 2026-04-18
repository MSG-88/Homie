"""Mesh events — the atom of distributed sync."""
from __future__ import annotations
import hashlib, json, os, time
from dataclasses import dataclass, field
from homie_core.utils import utc_now

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_LAST_TS = 0
_LAST_RANDOM = 0

def generate_ulid() -> str:
    global _LAST_TS, _LAST_RANDOM
    ts_ms = int(time.time() * 1000)
    if ts_ms == _LAST_TS:
        _LAST_RANDOM += 1
    else:
        _LAST_TS = ts_ms
        _LAST_RANDOM = int.from_bytes(os.urandom(10), "big")
    t = ts_ms
    ts_chars = []
    for _ in range(10):
        ts_chars.append(_CROCKFORD[t & 0x1F])
        t >>= 5
    ts_part = "".join(reversed(ts_chars))
    r = _LAST_RANDOM & ((1 << 80) - 1)
    rand_chars = []
    for _ in range(16):
        rand_chars.append(_CROCKFORD[r & 0x1F])
        r >>= 5
    rand_part = "".join(reversed(rand_chars))
    return ts_part + rand_part

def _compute_checksum(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

@dataclass
class HomieEvent:
    node_id: str
    category: str
    event_type: str
    payload: dict
    event_id: str = field(default_factory=generate_ulid)
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())
    vector_clock: dict = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = _compute_checksum(self.payload)

    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "node_id": self.node_id, "timestamp": self.timestamp,
                "category": self.category, "event_type": self.event_type, "payload": self.payload,
                "vector_clock": self.vector_clock, "checksum": self.checksum}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> HomieEvent:
        return cls(event_id=d["event_id"], node_id=d["node_id"], timestamp=d["timestamp"],
                   category=d["category"], event_type=d["event_type"], payload=d["payload"],
                   vector_clock=d.get("vector_clock", {}), checksum=d.get("checksum", ""))

    @classmethod
    def from_json(cls, raw: str) -> HomieEvent:
        return cls.from_dict(json.loads(raw))
