import time
import pytest
from unittest.mock import MagicMock, patch

from homie_core.vault.sync_manager import SyncManager
from homie_core.vault.models import ConnectionStatus


class TestSyncManagerTick:
    def test_tick_calls_sync_for_due_providers(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="always_on", sync_interval=300,
                last_sync=time.time() - 400,
            ),
        ]
        on_sync = MagicMock(return_value="synced 5 emails")
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 1
        assert results[0][0] == "gmail"
        on_sync.assert_called_once()

    def test_tick_skips_not_due_providers(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="always_on", sync_interval=300,
                last_sync=time.time() - 100,
            ),
        ]
        on_sync = MagicMock()
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 0
        on_sync.assert_not_called()

    def test_tick_skips_disconnected_providers(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=False,
                connection_mode="always_on", sync_interval=300,
            ),
        ]
        on_sync = MagicMock()
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 0

    def test_tick_skips_on_demand_providers(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="on_demand", sync_interval=300,
                last_sync=time.time() - 400,
            ),
        ]
        on_sync = MagicMock()
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 0

    def test_tick_handles_sync_error_gracefully(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="always_on", sync_interval=300,
                last_sync=time.time() - 400,
            ),
        ]
        on_sync = MagicMock(side_effect=Exception("API error"))
        manager = SyncManager(vault=vault, sync_callbacks={"gmail": on_sync})

        results = manager.tick()
        assert len(results) == 1
        assert "error" in results[0][1].lower()

    def test_tick_with_no_callback_registered(self):
        vault = MagicMock()
        vault.get_all_connections.return_value = [
            ConnectionStatus(
                provider="gmail", connected=True,
                connection_mode="always_on", sync_interval=300,
                last_sync=time.time() - 400,
            ),
        ]
        manager = SyncManager(vault=vault, sync_callbacks={})

        results = manager.tick()
        assert len(results) == 0
