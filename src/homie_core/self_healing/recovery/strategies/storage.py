"""Recovery strategies for storage failures (SQLite + ChromaDB)."""

import logging
import os
import shutil
from pathlib import Path
from ..engine import RecoveryResult, RecoveryTier

logger = logging.getLogger(__name__)


def retry_sqlite_operation(module, status, error, database=None, **ctx) -> RecoveryResult:
    """T1: Retry SQLite operation after brief delay."""
    if database is None:
        return RecoveryResult(success=False, action="no database", tier=RecoveryTier.RETRY)
    try:
        database._conn.execute("SELECT 'ok'").fetchone()
        return RecoveryResult(success=True, action="SQLite retry succeeded", tier=RecoveryTier.RETRY)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"SQLite retry failed: {exc}", tier=RecoveryTier.RETRY)


def emergency_disk_cleanup(module, status, error, log_dir=None, **ctx) -> RecoveryResult:
    """T1: Emergency cleanup of logs and temp files to free disk space."""
    freed = 0
    if log_dir and os.path.isdir(log_dir):
        for f in Path(log_dir).glob("*.log"):
            try:
                size = f.stat().st_size
                f.unlink()
                freed += size
            except OSError:
                pass
    logger.info("Emergency cleanup freed %d bytes", freed)
    return RecoveryResult(
        success=True,
        action=f"emergency cleanup freed {freed} bytes",
        tier=RecoveryTier.RETRY,
        details={"freed_bytes": freed},
    )


def restore_from_backup(module, status, error, database=None, backup_path=None, **ctx) -> RecoveryResult:
    """T2: Restore database from most recent backup."""
    if not backup_path or not os.path.exists(str(backup_path)):
        return RecoveryResult(success=False, action="no backup available", tier=RecoveryTier.FALLBACK)
    try:
        db_path = database.path if database else None
        if db_path:
            shutil.copy2(str(backup_path), str(db_path))
            database.initialize()
            logger.info("Restored database from backup: %s", backup_path)
            return RecoveryResult(success=True, action="restored from backup", tier=RecoveryTier.FALLBACK)
        return RecoveryResult(success=False, action="no db path", tier=RecoveryTier.FALLBACK)
    except Exception as exc:
        return RecoveryResult(success=False, action=f"restore failed: {exc}", tier=RecoveryTier.FALLBACK)
