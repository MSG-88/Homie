from __future__ import annotations

from typing import Optional

from homie_core.mesh.node_context import NodeContext


class ContextHandoff:
    """
    Detects when a user switches from one device (peer) to the local device,
    and emits a context handoff payload so Homie can offer continuity.
    """

    def __init__(self) -> None:
        self._last_active_peer: Optional[str] = None
        self._last_peer_context: Optional[NodeContext] = None
        self._handled_switches: set[tuple[str, str]] = set()

    def check(
        self,
        local_node_id: str,
        local_context: NodeContext,
        peer_contexts: dict[str, NodeContext],
    ) -> Optional[dict]:
        """
        Call on every context tick.

        Returns a handoff dict when a device-switch is detected, otherwise None.

        Handoff dict keys:
          from_node, to_node, previous_activity, previous_window,
          previous_minutes, message
        """
        # Find the most-active peer (if any)
        active_peer: Optional[str] = None
        active_peer_ctx: Optional[NodeContext] = None
        for pid, ctx in peer_contexts.items():
            if not ctx.is_idle and ctx.minutes_active > 0:
                if active_peer_ctx is None or ctx.minutes_active > active_peer_ctx.minutes_active:
                    active_peer, active_peer_ctx = pid, ctx

        # Phase 1: local is idle and a peer is active — remember the peer
        if active_peer and local_context.is_idle:
            self._last_active_peer = active_peer
            self._last_peer_context = active_peer_ctx
            return None

        # Phase 2: local just became active — check if the remembered peer went idle
        if (
            self._last_active_peer
            and self._last_peer_context
            and not local_context.is_idle
            and local_context.minutes_active > 0
        ):
            peer_id = self._last_active_peer
            peer_ctx = peer_contexts.get(peer_id)
            if peer_ctx and (peer_ctx.is_idle or peer_ctx.minutes_active == 0):
                key = (peer_id, local_node_id)
                if key not in self._handled_switches:
                    self._handled_switches.add(key)
                    saved = self._last_peer_context
                    # Clear state so we don't re-trigger
                    self._last_active_peer = None
                    self._last_peer_context = None
                    window_part = (
                        f" ({saved.active_window})" if saved.active_window else ""
                    )
                    return {
                        "from_node": peer_id,
                        "to_node": local_node_id,
                        "previous_activity": saved.activity_type,
                        "previous_window": saved.active_window,
                        "previous_minutes": saved.minutes_active,
                        "message": (
                            f"You were {saved.activity_type} on {saved.node_name}"
                            f"{window_part}. Want to continue here?"
                        ),
                    }

        return None
