# tests/unit/self_healing/test_boot_integration.py
import pytest
from unittest.mock import MagicMock, patch
from homie_core.self_healing.watchdog import HealthWatchdog


class TestBootIntegration:
    def test_watchdog_initializes_with_config(self, tmp_path):
        """Watchdog can be created from config."""
        wd = HealthWatchdog(db_path=tmp_path / "health.db", probe_interval=30.0)
        assert wd.system_health is not None

    def test_watchdog_probes_run_on_boot(self, tmp_path):
        """Probes can be run before main loop starts."""
        wd = HealthWatchdog(db_path=tmp_path / "health.db")
        results = wd.run_all_probes()
        # No probes registered, should return empty
        assert results == {}

    def test_watchdog_start_stop_lifecycle(self, tmp_path):
        """Watchdog starts and stops cleanly."""
        wd = HealthWatchdog(db_path=tmp_path / "health.db", probe_interval=0.1)
        wd.start()
        assert wd._running is True
        wd.stop()
        assert wd._running is False
