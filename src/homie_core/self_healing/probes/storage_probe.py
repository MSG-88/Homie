"""Health probe for SQLite and ChromaDB storage."""

import os
from .base import BaseProbe, HealthStatus, ProbeResult


class StorageProbe(BaseProbe):
    """Checks SQLite database and ChromaDB vector store health."""

    name = "storage"
    interval = 10.0  # critical

    def __init__(self, database, vector_store=None) -> None:
        self._db = database
        self._vectors = vector_store

    def check(self) -> ProbeResult:
        errors = []
        metadata = {}

        # Check SQLite
        try:
            result = self._db._conn.execute("SELECT 'ok'").fetchone()
            if result[0] != "ok":
                errors.append("SQLite query returned unexpected result")
            # Check DB file size
            if hasattr(self._db, "path") and os.path.exists(self._db.path):
                size_mb = os.path.getsize(self._db.path) / (1024 * 1024)
                metadata["db_size_mb"] = round(size_mb, 2)
        except Exception as exc:
            return ProbeResult(
                status=HealthStatus.FAILED,
                latency_ms=0,
                error_count=1,
                last_error=f"SQLite check failed: {exc}",
                metadata=metadata,
            )

        # Check ChromaDB
        vector_ok = True
        if self._vectors:
            try:
                self._vectors.query_episodes("health_check", n=1)
            except Exception as exc:
                vector_ok = False
                errors.append(f"ChromaDB: {exc}")

        status = HealthStatus.HEALTHY
        if not vector_ok:
            status = HealthStatus.DEGRADED

        return ProbeResult(
            status=status,
            latency_ms=0,
            error_count=len(errors),
            last_error=errors[-1] if errors else None,
            metadata=metadata,
        )
