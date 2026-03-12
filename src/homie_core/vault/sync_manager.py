"""SyncManager — periodic background sync for connected providers.

Called from the daemon's main-thread tick loop (every 60s). Checks which
always_on providers are due for sync and calls their registered callbacks.
On-demand providers are skipped (they sync only when explicitly triggered).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from homie_core.vault.models import ConnectionStatus


class SyncManager:
    """Manages periodic sync for connected providers."""

    def __init__(
        self,
        vault,
        sync_callbacks: Optional[dict[str, Callable]] = None,
    ):
        self._vault = vault
        self._callbacks: dict[str, Callable] = sync_callbacks or {}

    def register_callback(self, provider: str, callback: Callable) -> None:
        """Register a sync callback for a provider."""
        self._callbacks[provider] = callback

    def tick(self) -> list[tuple[str, str]]:
        """Check all connected providers and sync those that are due.

        Returns list of (provider, result_message) for providers that synced.
        Called from daemon's main loop every 60 seconds.
        """
        results: list[tuple[str, str]] = []

        try:
            connections = self._vault.get_all_connections()
        except Exception:
            return results

        now = time.time()

        for conn in connections:
            if not conn.connected:
                continue
            if conn.connection_mode != "always_on":
                continue
            if conn.provider not in self._callbacks:
                continue

            # Check if sync is due
            last = conn.last_sync or 0
            if (now - last) < conn.sync_interval:
                continue

            # Execute sync
            try:
                result = self._callbacks[conn.provider]()
                results.append((conn.provider, str(result)))
                # Update last_sync timestamp
                try:
                    self._vault.set_connection_status(
                        conn.provider,
                        connected=True,
                        label=conn.display_label,
                        mode=conn.connection_mode,
                        sync_interval=conn.sync_interval,
                        last_sync=time.time(),
                    )
                except Exception:
                    pass
            except Exception as e:
                error_msg = f"Sync error: {e}"
                results.append((conn.provider, error_msg))

        return results
