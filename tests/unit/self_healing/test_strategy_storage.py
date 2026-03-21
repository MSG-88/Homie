# tests/unit/self_healing/test_strategy_storage.py
import sqlite3
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.recovery.strategies.storage import (
    retry_sqlite_operation,
    emergency_disk_cleanup,
    restore_from_backup,
)
from homie_core.self_healing.recovery.engine import RecoveryResult, RecoveryTier


class TestStorageRecoveryStrategies:
    def test_retry_sqlite_succeeds_on_second_try(self):
        db = MagicMock()
        db._conn.execute.return_value.fetchone.return_value = ("ok",)
        result = retry_sqlite_operation(module="storage", status=2, error="locked", database=db)
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_retry_sqlite_fails(self):
        db = MagicMock()
        db._conn.execute.side_effect = sqlite3.OperationalError("still locked")
        result = retry_sqlite_operation(module="storage", status=2, error="locked", database=db)
        assert result.success is False

    def test_emergency_disk_cleanup(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "old.log").write_text("x" * 10000)
        result = emergency_disk_cleanup(module="storage", status=2, error="disk full", log_dir=str(log_dir))
        assert result.success is True
        assert result.tier == RecoveryTier.RETRY

    def test_restore_from_backup(self):
        db = MagicMock()
        result = restore_from_backup(module="storage", status=2, error="corrupt", database=db, backup_path="/fake")
        assert result.tier == RecoveryTier.FALLBACK
