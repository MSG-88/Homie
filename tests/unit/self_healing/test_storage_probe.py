# tests/unit/self_healing/test_storage_probe.py
import pytest
from unittest.mock import MagicMock
from homie_core.self_healing.probes.storage_probe import StorageProbe
from homie_core.self_healing.probes.base import HealthStatus


class TestStorageProbe:
    def test_healthy_when_db_works(self, tmp_path):
        db = MagicMock()
        db.path = tmp_path / "homie.db"
        db._conn = MagicMock()
        db._conn.execute.return_value.fetchone.return_value = ("ok",)
        vectors = MagicMock()
        probe = StorageProbe(database=db, vector_store=vectors)
        result = probe.check()
        assert result.status == HealthStatus.HEALTHY

    def test_degraded_when_vectors_fail(self, tmp_path):
        db = MagicMock()
        db.path = tmp_path / "homie.db"
        db._conn = MagicMock()
        db._conn.execute.return_value.fetchone.return_value = ("ok",)
        vectors = MagicMock()
        vectors.query_episodes.side_effect = RuntimeError("chroma down")
        probe = StorageProbe(database=db, vector_store=vectors)
        result = probe.check()
        assert result.status == HealthStatus.DEGRADED

    def test_failed_when_db_fails(self, tmp_path):
        db = MagicMock()
        db.path = tmp_path / "homie.db"
        db._conn = MagicMock()
        db._conn.execute.side_effect = Exception("corrupt")
        vectors = MagicMock()
        probe = StorageProbe(database=db, vector_store=vectors)
        result = probe.check()
        assert result.status == HealthStatus.FAILED
