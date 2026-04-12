"""Vector clock for causal ordering of distributed events."""
from __future__ import annotations


class VectorClock:
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
            if self._clocks.get(k, 0) > other._clocks.get(k, 0):
                return False
            if self._clocks.get(k, 0) < other._clocks.get(k, 0):
                at_least_one_less = True
        return at_least_one_less

    def is_concurrent(self, other: VectorClock) -> bool:
        return not self.happens_before(other) and not other.happens_before(self)

    def to_dict(self) -> dict[str, int]:
        return dict(self._clocks)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> VectorClock:
        return cls(d)
