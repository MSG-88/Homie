import pytest
from unittest.mock import MagicMock
from homie_core.model_evolution.model_registry import ModelRegistry, ModelVersion


class TestModelVersion:
    def test_creation(self):
        v = ModelVersion(
            version_id="homie-v1",
            base_model="lfm2",
            ollama_name="MSG-88/Homie",
            modelfile_hash="abc123",
            status="active",
        )
        assert v.version_id == "homie-v1"
        assert v.is_active is True

    def test_to_dict(self):
        v = ModelVersion(version_id="v1", base_model="lfm2", ollama_name="test", modelfile_hash="x")
        d = v.to_dict()
        assert d["version_id"] == "v1"


class TestModelRegistry:
    def test_register_version(self):
        storage = MagicMock()
        reg = ModelRegistry(storage=storage)
        version = reg.register("lfm2", "MSG-88/Homie", "hash123", changelog="Initial version")
        assert version.version_id.startswith("homie-v")
        storage.save_model_version.assert_called()

    def test_get_active_version(self):
        storage = MagicMock()
        storage.get_active_model_version.return_value = {"version_id": "homie-v1", "status": "active", "base_model": "lfm2", "ollama_name": "test", "modelfile_hash": "x", "metrics": "{}", "changelog": "init"}
        reg = ModelRegistry(storage=storage)
        active = reg.get_active()
        assert active is not None
        assert active.version_id == "homie-v1"

    def test_promote_version(self):
        storage = MagicMock()
        reg = ModelRegistry(storage=storage)
        reg.promote("homie-v2")
        storage.update_model_version_status.assert_called_with("homie-v2", "active")

    def test_rollback(self):
        storage = MagicMock()
        storage.get_previous_model_version.return_value = {"version_id": "homie-v1", "status": "archived", "base_model": "lfm2", "ollama_name": "test", "modelfile_hash": "x", "metrics": "{}", "changelog": ""}
        reg = ModelRegistry(storage=storage)
        prev = reg.rollback("homie-v2")
        assert prev is not None
        storage.update_model_version_status.assert_any_call("homie-v2", "rolled_back")

    def test_list_versions(self):
        storage = MagicMock()
        storage.list_model_versions.return_value = [
            {"version_id": "v1", "status": "archived", "base_model": "lfm2", "ollama_name": "t", "modelfile_hash": "x", "metrics": "{}", "changelog": ""},
            {"version_id": "v2", "status": "active", "base_model": "lfm2", "ollama_name": "t", "modelfile_hash": "y", "metrics": "{}", "changelog": ""},
        ]
        reg = ModelRegistry(storage=storage)
        versions = reg.list_versions()
        assert len(versions) == 2
