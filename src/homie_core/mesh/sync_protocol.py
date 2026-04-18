"""Delta sync protocol — exchange events between Hub and Spokes."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore


@dataclass
class SyncRequest:
    node_id: str
    last_event_id: Optional[str]
    vector_clock: dict = field(default_factory=dict)


@dataclass
class SyncResponse:
    events: list[HomieEvent] = field(default_factory=list)
    hub_event_id: Optional[str] = None


class SyncProtocol:
    def __init__(self, node_id: str, event_store: EventStore):
        self._node_id = node_id
        self._store = event_store

    def prepare_response(self, request: SyncRequest) -> SyncResponse:
        events = self._store.events_since(request.last_event_id, limit=5000)
        return SyncResponse(events=events, hub_event_id=self._store.last_event_id())

    def apply_response(self, response: SyncResponse) -> int:
        applied = 0
        for event in response.events:
            if self._store.get(event.event_id) is None:
                self._store.append(event)
                applied += 1
        return applied

    def get_unsynced_events(self, limit: int = 5000) -> list[HomieEvent]:
        return self._store.unsynced_events(limit=limit)

    def mark_events_synced(self, event_ids: list[str]) -> None:
        self._store.mark_synced(event_ids)

    def prepare_request(self) -> SyncRequest:
        return SyncRequest(
            node_id=self._node_id,
            last_event_id=self._store.last_event_id(),
            vector_clock={},
        )
