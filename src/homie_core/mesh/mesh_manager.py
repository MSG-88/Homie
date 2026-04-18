"""Mesh manager — top-level coordinator for mesh operations."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from homie_core.mesh.events import HomieEvent
from homie_core.mesh.event_store import EventStore
from homie_core.mesh.sync_protocol import SyncProtocol, SyncRequest, SyncResponse
from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.vector_clock import VectorClock


class MeshManager:
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

    def prepare_sync_request(self) -> SyncRequest:
        return self._sync.prepare_request()

    def handle_sync_request(self, request: SyncRequest) -> SyncResponse:
        return self._sync.prepare_response(request)

    def apply_sync_response(self, response: SyncResponse) -> int:
        applied = self._sync.apply_response(response)
        # Events received from hub are already synced — mark them so they
        # don't appear in get_unsynced_for_hub() as events to push back.
        synced_ids = [e.event_id for e in response.events]
        if synced_ids:
            self._sync.mark_events_synced(synced_ids)
        return applied

    def get_unsynced_for_hub(self, limit: int = 5000) -> list[HomieEvent]:
        return self._sync.get_unsynced_events(limit=limit)

    def mark_pushed_to_hub(self, event_ids: list[str]) -> None:
        self._sync.mark_events_synced(event_ids)
